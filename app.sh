#!/usr/bin/env bash

# 1. 强行进入当前脚本所在的目录（防止路径错乱）
cd "$(dirname "$0")"

# 2. 打印当前启动日志，方便在守护进程的 daemon.log 里观察
echo "==== 正在启动 FastAPI 应用 ===="

# 3. 使用 exec 启动 Uvicorn 服务
#    - VENV_BIN 是守护进程传过来的环境变量，确保直接调用虚拟环境内的 uvicorn，防止找不到命令
#    - main:app 代表执行 main.py 里的 app = FastAPI() 对象（根据你实际的文件名修改）
#    - --host 0.0.0.0 让服务监听所有网卡
#    - --port 8000 是你网页服务运行的端口
exec "${VENV_BIN}/uvicorn" main:app --host 0.0.0.0 --port 8000