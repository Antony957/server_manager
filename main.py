"""
main.py
本机控制台 + 反向代理服务 (完全配置驱动与动态并行管理版)

启动: uvicorn main:app --host 0.0.0.0 --port 9000 --reload
"""
import asyncio
import json
import os
import yaml
import copy
from pathlib import Path

import asyncssh
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ssh_manager import SSHManager

BASE_DIR = Path(__file__).parent

def load_config():
    """从 yaml 文件中加载全局服务器以及公共模型模板，并完成独立内存灌入"""
    models_path = os.path.join(BASE_DIR, "models.yaml")
    servers_path = os.path.join(BASE_DIR, "servers.yaml")

    with open(models_path, "r", encoding="utf-8") as f:
        shared_models = yaml.safe_load(f).get("models", [])

    with open(servers_path, "r", encoding="utf-8") as f:
        servers_data = yaml.safe_load(f)

    # 深度拷贝模型，确保各服务器节点并行参数、参数隔离，支持运行时动态肆意改写
    for srv in servers_data.get("servers", []):
        srv["models"] = copy.deepcopy(shared_models)

    return servers_data

# === 🚀 全局变量及状态机初始化 ===
CONFIG = load_config()
CONSOLE_PORT = CONFIG.get("console_port", 9000)

SERVERS = {s["id"]: s for s in CONFIG["servers"]}
MANAGERS = {sid: SSHManager(cfg) for sid, cfg in SERVERS.items()}

# 全局状态字典：状态精简为最纯粹的 running / stopped
STATE = {
    "active_server": None,
    "server_status": {sid: "stopped" for sid in SERVERS},  # stopped / running / error
    "last_model_id": {sid: None for sid in SERVERS},      # 记录当前运行的模型 id
}

app = FastAPI(title="vLLM 智能控制台")

# ---------------------------------------------------------------------------
# 内部状态管理辅助工具
# ---------------------------------------------------------------------------
def _get_active_model_port(server_id: str) -> int:
    """动态获取某台服务器上当前正在运行或默认准备运行的模型监听端口"""
    server_cfg = SERVERS[server_id]
    current_mid = STATE["last_model_id"].get(server_id)

    # 优先寻找正在运行的模型端口
    if current_mid:
        for m in server_cfg.get("models", []):
            if m["id"] == current_mid:
                return m["vllm_config"].get("port", 33261)

    # 没有运行则默认拿第一个模型作为兜底探活端口
    if server_cfg.get("models"):
        return server_cfg["models"][0]["vllm_config"].get("port", 33261)
    return 33261

# ---------------------------------------------------------------------------
# 后台核心管理 API
# ---------------------------------------------------------------------------

@app.get("/api/servers")
async def list_servers():
    """下发全量状态给前端，包括注入的 models 静态参数列表"""
    result = []
    for sid, cfg in SERVERS.items():
        # 实时动态修正当前探活或运转需要指向的端口
        current_port = _get_active_model_port(sid)

        # 组装当前的激活模型名字
        running_model_name = "-"
        if STATE["last_model_id"][sid]:
            m_obj = next((m for m in cfg["models"] if m["id"] == STATE["last_model_id"][sid]), None)
            if m_obj:
                running_model_name = m_obj["name"]

        result.append({
            "id":            sid,
            "name":          cfg["name"],
            "host":          cfg["host"],
            "vllm_port":     current_port,
            "status":        STATE["server_status"][sid],
            "is_active":     STATE["active_server"] == sid,
            "current_model": running_model_name,
            "models":        cfg["models"]  # 把写死的模型列表下发给前端下拉框渲染
        })
    return {"servers": result, "active_server": STATE["active_server"]}


@app.get("/api/servers/{server_id}/gpus")
async def get_gpus(server_id: str):
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")
    gpus = await asyncio.to_thread(MANAGERS[server_id].get_gpu_status)
    return {"server_id": server_id, "gpus": gpus}


@app.get("/api/gpus")
async def get_all_gpus():
    async def fetch(sid):
        gpus = await asyncio.to_thread(MANAGERS[sid].get_gpu_status)
        return sid, gpus
    results = await asyncio.gather(*[fetch(sid) for sid in SERVERS], return_exceptions=True)
    out = {}
    for item in results:
        if isinstance(item, Exception): continue
        sid, gpus = item
        out[sid] = gpus
    return out


@app.post("/api/servers/{server_id}/start")
async def start_server(server_id: str, payload: dict):
    """
    启动/切换服务接口
    接收 payload: {"model_index": 0, "gpus": [0, 1]}
    """
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")

    cfg = SERVERS[server_id]
    model_idx = payload.get("model_index", 0)
    gpus = payload.get("gpus", [])

    if model_idx >= len(cfg["models"]):
        raise HTTPException(400, "非法的模型索引参数")

    # 1. 抓取选中的目标公共模型字典（已深拷贝完成）
    target_model = cfg["models"][model_idx]
    custom_tp = payload.get("tensor_parallel_size") or target_model.get("tensor_parallel_size", 1)
    custom_pp = payload.get("pipeline_parallel_size") or target_model.get("pipeline_parallel_size", 1)

    target_model["tensor_parallel_size"] = int(custom_tp)
    target_model["pipeline_parallel_size"] = int(custom_pp)

    if gpus and len(gpus) > 0:
        target_model["tensor_parallel_size"] = int(custom_tp)
        target_model["pipeline_parallel_size"] = int(custom_pp)

    print(f"[START] 准备部署节点: {server_id} | 模型: {target_model['name']} | TP: {target_model['tensor_parallel_size']} | PP: {target_model['pipeline_parallel_size']} | GPUs: {gpus}")

    # 2. 互斥锁：先礼后兵强杀旧的活跃节点
    old_active = STATE["active_server"]
    if old_active and old_active != server_id:
        try:
            print(f"[START] 正在腾挪资源，优雅强杀旧节点 {old_active}...")
            await asyncio.to_thread(MANAGERS[old_active].stop_vllm, True)
            STATE["server_status"][old_active] = "stopped"
            STATE["last_model_id"][old_active]  = None
        except Exception as e:
            print(f"[START] 旧节点清理时发生非致命抖动: {e}")
            STATE["server_status"][old_active] = "stopped"

    # 3. 记录运行时状态，拉起新节点
    STATE["last_model_id"][server_id] = target_model["id"]
    STATE["active_server"] = server_id
    # 注意：前端只保留运行中和已停止。在就绪前，我们在后台维持状态，直到异步轮询将它变绿
    STATE["server_status"][server_id] = "stopped"

    try:
        mgr = MANAGERS[server_id]
        # 丢进线程池，将定制好的完整参数字典 target_model 扔进大管家去远程生成 JSON/YAML 配置并启动
        res = await asyncio.to_thread(mgr.start_vllm, target_model, gpus)
        print(f"[START] 远程命令下发完毕。结果反馈: {res}")
        STATE["server_status"][server_id] = "running"

        # 4. 激活异步双重探活守护协程，绝不阻塞主线程

        return {"ok": True, "ssh_result": res}
    except Exception as e:
        print(f"[START] 发生灾难性拉起异常: {e}")
        STATE["server_status"][server_id] = "stopped"
        STATE["active_server"] = None
        STATE["last_model_id"][server_id] = None
        raise HTTPException(500, f"部署链条中断: {e}")


@app.post("/api/servers/{server_id}/stop")
async def stop_server(server_id: str):
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")
    try:
        # 调用基于 pkill -f 的干净收尸函数
        await asyncio.to_thread(MANAGERS[server_id].stop_vllm, True)
        STATE["server_status"][server_id] = "stopped"
        STATE["last_model_id"][server_id]  = None
        if STATE["active_server"] == server_id:
            STATE["active_server"] = None
        return {"ok": True}
    except Exception as e:
        STATE["server_status"][server_id] = "stopped"
        raise HTTPException(500, f"停止失败: {e}")


@app.get("/api/servers/{server_id}/logs/stream")
async def stream_log(server_id: str):
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")
    cfg = SERVERS[server_id]

    async def generator():
        conn = None
        process = None
        try:
            conn = await asyncssh.connect(
                cfg["host"],
                port=cfg.get("ssh_port", 22),
                username=cfg["ssh_user"],
                client_keys=[cfg["ssh_key_path"]] if cfg.get("ssh_key_path") else None,
                password=cfg.get("ssh_password") or None,
                known_hosts=None,
            )
            process = await conn.create_process(f"tail -f {cfg['log_path']}")

            async for line in process.stdout:
                yield f"data: {line}\n\n"

        except (asyncio.CancelledError, GeneratorExit):
            print(f"[🧹 清理] 收到退出或重载信号，瞬间强制断开服务器 {server_id} 的实时日志 SSH 通道...")
            raise
        except Exception as e:
            yield f"data: [日志串流中断: {e}]\n\n"
        finally:
            if process:
                try: process.terminate()
                except: pass
            if conn:
                conn.close()
                await conn.wait_closed()
                print(f"[🧹 清理] 服务器 {server_id} 日志串流资源完整回收。")

    return StreamingResponse(generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# OpenAI 智能兼容反向代理层（路由端口自适应）
# ---------------------------------------------------------------------------
@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    active = STATE["active_server"]
    if not active:
        raise HTTPException(503, "当前无活跃的大模型实例在提供服务，请先去控制台一键启动。")
    if STATE["server_status"][active] != "running":
        raise HTTPException(503, "当前大模型实例正在初始化权重/KV缓存中，请稍候...")

    cfg = SERVERS[active]
    current_runtime_port = _get_active_model_port(active)

    target_url = f"http://{cfg['host']}:{current_runtime_port}/v1/{path}"

    body = await request.body()
    if request.method == "POST":
        try:
            body_snippet = body.decode("utf-8")[:200]
        except Exception:
            print("[📦 请求内容] Body 包含无法解析的二进制数据")

    # 2. 过滤并提取 Headers
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    # 3. 注入 Token 鉴权
    current_mid = STATE["last_model_id"].get(active)
    target_model = next((m for m in cfg.get("models", []) if m["id"] == current_mid), None)

    if target_model and "vllm_config" in target_model:
        v_cfg = target_model["vllm_config"]
        config_api_key = v_cfg.get("api-key")
        if config_api_key:
            headers["Authorization"] = f"Bearer {config_api_key}"


    timeout_cfg = httpx.Timeout(connect=10, read=600, write=60, pool=10)

    # 4. 用安全上下文管理器拉起底层转发连接
    async with httpx.AsyncClient(timeout=timeout_cfg, trust_env=False) as client:

        # --- 处理 GET 分支 ---
        if request.method == "GET":
            print("[📡 发送上游] 正在执行 GET 同步请求...")
            try:
                r = await client.get(target_url, headers=headers, params=request.query_params)
                return JSONResponse(content=r.json(), status_code=r.status_code)
            except Exception as e:
                raise HTTPException(500, f"上游连接失败: {e}")

        # --- 处理 POST 分支 ---
        try:
            req = client.build_request("POST", target_url, headers=headers, content=body)
            upstream = await client.send(req, stream=True)
        except Exception as e:
            raise HTTPException(500, f"上游建立失败: {e}")

        # 检查是否命中大模型标准的 text/event-stream 流式
        if "text/event-stream" in upstream.headers.get("content-type", ""):
            async def event_gen():
                chunk_count = 0
                try:
                    async for chunk in upstream.aiter_raw():
                        chunk_count += 1
                        # 每拿到10个数据块在后台打印一下进度，避免疯狂刷屏
                        if chunk_count % 10 == 0:
                            print(f"[⏳ 流式代理中] 已安全转发 {chunk_count} 个数据块到前端...")
                        yield chunk
                except Exception as stream_err:
                    print(f"[💥 流中断] 流式迭代过程中上游或前端发生断开: {stream_err}")
                finally:
                    print(f"[✅ 流结束] 异步生成器退出，总计转发 {chunk_count} 个数据块，正在关闭上游句柄")
                    await upstream.aclose()

            return StreamingResponse(event_gen(), media_type="text/event-stream", status_code=upstream.status_code)

        else:
            print("[🗂️ 分支判定] 未检测到流式标识，切入【普通 JSON 代理】通道")
            try:
                content = await upstream.aread()
                await upstream.aclose()
                print(f"[✅ JSON 响应完结] 成功读取全部内容，大小: {len(content)} bytes")
                return JSONResponse(content=json.loads(content) if content else {}, status_code=upstream.status_code)
            except Exception as json_err:
                print(f"[💥 读取中断] 读取非流式 JSON 响应体失败: {json_err}")
                await upstream.aclose()
                raise HTTPException(500, f"读取响应失败: {json_err}")


# @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
# async def proxy(path: str, request: Request):
#     active = STATE["active_server"]
#     if not active:
#         raise HTTPException(503, "当前无活跃的大模型实例在提供服务，请先去控制台一键启动。")
#     if STATE["server_status"][active] != "running":
#         raise HTTPException(503, "当前大模型实例正在初始化权重/KV缓存中，请稍候...")
#
#     cfg = SERVERS[active]
#     # 核心联动：反向代理时，必须实时捕获该激活模型在 models.yaml 中配置的真实运行端口
#     current_runtime_port = _get_active_model_port(active)
#
#     target_url = f"http://{cfg['host']}:{current_runtime_port}/v1/{path}"
#     body = await request.body()
#     headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
#
#     # trust_env=False 彻底隔绝本地代理工具的502大坑，直连物理机网卡
#     client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10), trust_env=False)
#
#     if request.method == "GET":
#         r = await client.get(target_url, headers=headers, params=request.query_params)
#         await client.aclose()
#         return JSONResponse(content=r.json(), status_code=r.status_code)
#
#     req = client.build_request("POST", target_url, headers=headers, content=body)
#     upstream = await client.send(req, stream=True)
#
#     if "text/event-stream" in upstream.headers.get("content-type", ""):
#         async def event_gen():
#             try:
#                 async for chunk in upstream.aiter_raw():
#                     yield chunk
#             finally:
#                 await upstream.aclose()
#                 await client.aclose()
#         return StreamingResponse(event_gen(), media_type="text/event-stream", status_code=upstream.status_code)
#     else:
#         content = await upstream.aread()
#         await upstream.aclose()
#         await client.aclose()
#         return JSONResponse(content=json.loads(content) if content else {}, status_code=upstream.status_code)

# ---------------------------------------------------------------------------
# 静态前端挂载
# ---------------------------------------------------------------------------
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")