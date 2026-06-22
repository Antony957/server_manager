"""
test_proxy.py
用于测试 FastAPI 控制台反向代理路由的安全探活与流式流传输。
运行前请确保：FastAPI 控制台已在本地 9000 端口启动，且有一台 vLLM 服务器处于 running 状态。

运行命令: python test_proxy.py
"""
import json
import http.client
import urllib.parse

CONSOLE_HOST = "127.0.0.1"
CONSOLE_PORT = 9000


def test_get_models():
    """测试 1: 验证 /v1/models 接口是否能正常反代并获取模型列表"""
    print("\n==================================================")
    print("🎬 正在测试 [GET] /v1/models (模型列表反代)...")
    print("==================================================")

    conn = http.client.HTTPConnection(CONSOLE_HOST, CONSOLE_PORT, timeout=10)
    try:
        # 注意：这里我们故意不带任何 Authorization Header，测试反代是否会自动注入！
        conn.request("GET", "/v1/models")
        response = conn.getresponse()

        print(f"统计状态码: {response.status}")
        data = response.read().decode("utf-8")

        if response.status == 200:
            print("🟢 成功收到 200 响应！内容如下：")
            parsed = json.loads(data)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        else:
            print(f"🔴 接口未能正常返回 200。错误信息:\n{data}")

    except Exception as e:
        print(f"💥 网络请求发生异常: {e}")
    finally:
        conn.close()


def test_post_chat_stream():
    """测试 2: 验证 /v1/chat/completions 是否能完美支撑 text/event-stream 流式传输"""
    print("\n==================================================")
    print("🎬 正在测试 [POST] /v1/chat/completions (大模型流式对话)...")
    print("==================================================")

    # 模拟标准的 OpenAI 聊天请求体
    payload = {
        "model": "Qwen/Qwen2-VL-7B-Instruct",  # 这里传什么都行，反代会自适应转发
        "messages": [
            {"role": "user", "content": "你好！请用一句话证明你已经成功通过了反向代理并跑起来了。"}
        ],
        "stream": True  # 开启流式传输
    }

    body_data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        # 同样故意不写 Token，测试网关智能补全
    }

    conn = http.client.HTTPConnection(CONSOLE_HOST, CONSOLE_PORT, timeout=600)
    try:
        conn.request("POST", "/v1/chat/completions", body=body_data, headers=headers)
        response = conn.getresponse()

        print(f"统计状态码: {response.status}")
        print(f"Content-Type 响应头: {response.getheader('Content-Type')}\n")

        if response.status != 200:
            print(f"🔴 转发流失败。错误原因:\n{response.read().decode('utf-8')}")
            return

        print("🟢 成功建立 SSE 串流连接！开始接收实时生成文本：\n" + "-" * 40)

        # 逐行读取流式套接字，模拟打字机回显
        while True:
            line_bytes = response.readline()
            if not line_bytes:
                break  # 流结束了

            line = line_bytes.decode("utf-8").strip()

            # 过滤标准 SSE 的 data: 标签
            if line.startswith("data:"):
                json_str = line[5:].strip()

                # 忽略 OpenAI 规范的结束标志 data: [DONE]
                if json_str == "[DONE]":
                    print("\n" + "-" * 40 + "\n🏁 [流式传输正常结束]")
                    break

                try:
                    chunk = json.loads(json_str)
                    # 顺着 OpenAI 格式提取 delta 文本
                    delta_text = chunk["choices"][0]["delta"].get("content", "")
                    # 流式打印到屏幕，不换行，立刻刷新缓冲区
                    print(delta_text, end="", flush=True)
                except Exception:
                    # 容错：如果是其他非标准块，原样吐出
                    pass

    except Exception as e:
        print(f"💥 流式连接捕获到崩溃: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    # 执行双重功能测试
    test_get_models()
    test_post_chat_stream()