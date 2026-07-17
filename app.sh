#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# LOCAL_PORT 由守护进程注入（config.json 中为 9000）；venv 的 bin 已在 PATH 中，也可直接用 $VENV_BIN
exec uvicorn main:app --host 127.0.0.1 --port "${LOCAL_PORT:-8000}"