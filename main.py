"""
main.py
本机控制台 + 反向代理服务

启动: uvicorn main:app --host 0.0.0.0 --port 9000
"""
import asyncio
import json
import yaml
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx

from ssh_manager import SSHManager

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SERVERS = {s["id"]: s for s in CONFIG["servers"]}
MANAGERS = {sid: SSHManager(cfg) for sid, cfg in SERVERS.items()}

# 全局状态(进程内存,简单场景够用;要持久化可换成文件/sqlite)
STATE = {
    "active_server": None,   # 当前选中并对外提供服务的 server id
    "server_status": {sid: "stopped" for sid in SERVERS},  # stopped/starting/running/stopping/error
    "last_model": {sid: None for sid in SERVERS},
}

app = FastAPI(title="vLLM 控制台")

# ---------------------------------------------------------------------------
# 管理 API
# ---------------------------------------------------------------------------

@app.get("/api/servers")
async def list_servers():
    """返回4台服务器的基本信息+当前状态(附带一次轻量健康检查)"""
    result = []
    for sid, cfg in SERVERS.items():
        mgr = MANAGERS[sid]
        healthy = await mgr.health_check()
        status = STATE["server_status"][sid]
        if status == "running" and not healthy:
            status = "error"  # 标记为已起进程但接口没通
            STATE["server_status"][sid] = status
        result.append({
            "id": sid,
            "name": cfg["name"],
            "host": cfg["host"],
            "vllm_port": cfg["vllm_port"],
            "default_model": cfg["default_model"],
            "default_extra_args": cfg["default_extra_args"],
            "status": status,
            "healthy": healthy,
            "is_active": STATE["active_server"] == sid,
            "current_model": STATE["last_model"][sid],
        })
    return {"servers": result, "active_server": STATE["active_server"]}


@app.get("/api/servers/{server_id}/gpus")
async def get_gpus(server_id: str):
    """查询某台服务器各GPU的显存/利用率/进程占用情况,用于决定开哪几张卡"""
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")
    gpus = await asyncio.to_thread(MANAGERS[server_id].get_gpu_status)
    return {"server_id": server_id, "gpus": gpus}


@app.get("/api/gpus")
async def get_all_gpus():
    """一次性查询4台服务器的GPU状态(并发SSH查询,前端做一个总览大盘)"""
    async def fetch(sid):
        gpus = await asyncio.to_thread(MANAGERS[sid].get_gpu_status)
        return sid, gpus
    results = await asyncio.gather(*[fetch(sid) for sid in SERVERS])
    return {sid: gpus for sid, gpus in results}


@app.post("/api/servers/{server_id}/start")
async def start_server(server_id: str, payload: dict):
    """
    启动指定服务器的 vLLM,并将其设为当前激活服务器(自动关闭旧的激活服务器)。
    payload: { "model": "...", "extra_args": "...", "port": 8000(可选) }
    """
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")

    cfg = SERVERS[server_id]
    model = payload.get("model") or cfg["default_model"]
    extra_args = payload.get("extra_args", cfg["default_extra_args"])
    port = payload.get("port") or cfg["vllm_port"]
    gpus = payload.get("gpus")  # 例如 [0,1],为空则用全部卡

    # 若指定了多卡且 extra_args 里没写 tensor-parallel-size,自动补上,免得选了2张卡却只用1张
    if gpus and len(gpus) > 1 and "--tensor-parallel-size" not in extra_args:
        extra_args = f"{extra_args} --tensor-parallel-size {len(gpus)}".strip()

    # 1. 如果有其他服务器在跑,先关掉(同一时间只允许1台)
    old_active = STATE["active_server"]
    if old_active and old_active != server_id:
        STATE["server_status"][old_active] = "stopping"
        try:
            MANAGERS[old_active].stop_vllm(ignore_errors=True)
            STATE["server_status"][old_active] = "stopped"
            STATE["last_model"][old_active] = None
        except Exception as e:
            STATE["server_status"][old_active] = "error"

    # 2. 启动目标服务器
    STATE["server_status"][server_id] = "starting"
    try:
        mgr = MANAGERS[server_id]
        res = mgr.start_vllm(model=model, port=port, extra_args=extra_args, gpus=gpus)
        STATE["last_model"][server_id] = model
        STATE["active_server"] = server_id
        # 注:vLLM 加载模型可能要几十秒到几分钟,这里不阻塞等待,
        # 前端轮询 /api/servers 的 healthy 字段判断是否真正就绪
        asyncio.create_task(_mark_running_when_healthy(server_id))
        return {"ok": True, "message": "已下发启动命令,模型加载中,请通过状态轮询确认就绪", "ssh_result": res}
    except Exception as e:
        STATE["server_status"][server_id] = "error"
        STATE["active_server"] = None
        raise HTTPException(500, f"启动失败: {e}")


async def _mark_running_when_healthy(server_id: str, timeout: int = 600, interval: int = 5):
    """后台轮询健康检查,就绪后把状态从 starting 改为 running"""
    mgr = MANAGERS[server_id]
    waited = 0
    while waited < timeout:
        if STATE["active_server"] != server_id:
            return  # 中途被切换走了,放弃
        if await mgr.health_check():
            STATE["server_status"][server_id] = "running"
            return
        await asyncio.sleep(interval)
        waited += interval
    if STATE["active_server"] == server_id:
        STATE["server_status"][server_id] = "error"


@app.post("/api/servers/{server_id}/stop")
async def stop_server(server_id: str):
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")
    STATE["server_status"][server_id] = "stopping"
    try:
        MANAGERS[server_id].stop_vllm(ignore_errors=True)
        STATE["server_status"][server_id] = "stopped"
        STATE["last_model"][server_id] = None
        if STATE["active_server"] == server_id:
            STATE["active_server"] = None
        return {"ok": True}
    except Exception as e:
        STATE["server_status"][server_id] = "error"
        raise HTTPException(500, f"停止失败: {e}")


@app.get("/api/servers/{server_id}/logs")
async def get_logs(server_id: str, n: int = 200):
    if server_id not in SERVERS:
        raise HTTPException(404, "未知服务器")
    log = MANAGERS[server_id].tail_log(n)
    return {"log": log}


# ---------------------------------------------------------------------------
# OpenAI 兼容反向代理: 所有 /v1/* 请求转发到当前激活服务器
# ---------------------------------------------------------------------------

@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    active = STATE["active_server"]
    if not active:
        raise HTTPException(503, "当前没有激活的 vLLM 服务器,请先在控制台启动一台")
    if STATE["server_status"][active] not in ("running",):
        raise HTTPException(503, f"目标服务器状态为 {STATE['server_status'][active]},尚未就绪")

    cfg = SERVERS[active]
    target_url = f"http://{cfg['host']}:{cfg['vllm_port']}/v1/{path}"
    body = await request.body()
    is_stream = b'"stream"' in body and b'"stream": true' in body.replace(b" ", b"") or b'"stream":true' in body.replace(b" ", b"")

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10))

    if request.method == "GET":
        r = await client.get(target_url, headers=headers, params=request.query_params)
        await client.aclose()
        return JSONResponse(content=r.json(), status_code=r.status_code)

    # POST: 判断是否流式(SSE)
    req = client.build_request("POST", target_url, headers=headers, content=body)
    upstream = await client.send(req, stream=True)

    if "text/event-stream" in upstream.headers.get("content-type", ""):
        async def event_gen():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
        return StreamingResponse(event_gen(), media_type="text/event-stream",
                                  status_code=upstream.status_code)
    else:
        content = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return JSONResponse(content=json.loads(content) if content else {},
                             status_code=upstream.status_code)


# ---------------------------------------------------------------------------
# 静态前端
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
