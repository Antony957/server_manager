"""
ssh_manager.py
通过 SSH 在远程服务器上启停 vLLM,并探活。
"""
import shlex
import paramiko
import httpx


class SSHManager:
    def __init__(self, server_cfg: dict):
        self.cfg = server_cfg

    def _connect(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = dict(
            hostname=self.cfg["host"],
            port=self.cfg.get("ssh_port", 22),
            username=self.cfg["ssh_user"],
            timeout=10,
        )
        if self.cfg.get("ssh_key_path"):
            kwargs["key_filename"] = self.cfg["ssh_key_path"]
        if self.cfg.get("ssh_password"):
            kwargs["password"] = self.cfg["ssh_password"]
        client.connect(**kwargs)
        return client

    def _exec(self, command: str) -> tuple[int, str, str]:
        client = self._connect()
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=30)
            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode(errors="ignore")
            err = stderr.read().decode(errors="ignore")
            return exit_code, out, err
        finally:
            client.close()

    def is_process_alive(self) -> bool:
        """通过 pidfile 检查远程进程是否还活着"""
        pid_path = self.cfg["pid_path"]
        cmd = f"test -f {pid_path} && kill -0 $(cat {pid_path}) 2>/dev/null && echo ALIVE || echo DEAD"
        try:
            _, out, _ = self._exec(cmd)
            return "ALIVE" in out
        except Exception:
            return False

    def start_vllm(self, model: str, port: int, extra_args: str = "", gpus: list[int] | None = None) -> dict:
        """
        后台启动 vLLM:
        CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve <model> --port <port> [extra_args] > log 2>&1 &
        echo $! > pidfile
        gpus: 要使用的GPU index列表,例如 [0,1];为空则不设置(用全部卡)
        """
        log_path = self.cfg["log_path"]
        pid_path = self.cfg["pid_path"]
        model_q = shlex.quote(model)

        # 先确保旧进程已停(防止端口/显存占用)
        self.stop_vllm(ignore_errors=True)

        env_prefix = ""
        if gpus:
            env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in gpus)} "

        cmd = (
            f"{env_prefix}nohup vllm serve {model_q} --port {port} --host 0.0.0.0 {extra_args} "
            f"> {log_path} 2>&1 & echo $! > {pid_path}"
        )
        full_cmd = f"bash -lc {shlex.quote(cmd)}"
        exit_code, out, err = self._exec(full_cmd)
        return {"exit_code": exit_code, "stdout": out, "stderr": err}

    def stop_vllm(self, ignore_errors: bool = False) -> dict:
        pid_path = self.cfg["pid_path"]
        cmd = (
            f"if [ -f {pid_path} ]; then "
            f"kill -9 $(cat {pid_path}) 2>/dev/null; rm -f {pid_path}; fi"
        )
        try:
            exit_code, out, err = self._exec(f"bash -lc {shlex.quote(cmd)}")
            return {"exit_code": exit_code, "stdout": out, "stderr": err}
        except Exception as e:
            if ignore_errors:
                return {"exit_code": -1, "stdout": "", "stderr": str(e)}
            raise

    def get_gpu_status(self) -> list[dict]:
        """
        用 nvidia-smi 查询每张卡的状态(比解析 nvitop TUI 输出更稳定)。
        返回: [{index, name, mem_used_mb, mem_total_mb, util_pct, temp_c, processes:[{pid, name, mem_mb}]}, ...]
        """
        query = (
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
        )
        proc_query = (
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
            "--format=csv,noheader,nounits"
        )
        uuid_query = "nvidia-smi --query-gpu=index,uuid --format=csv,noheader"

        try:
            _, gpu_out, _ = self._exec(query)
            _, proc_out, _ = self._exec(proc_query)
            _, uuid_out, _ = self._exec(uuid_query)
        except Exception as e:
            return [{"error": str(e)}]

        # uuid -> index 映射,用于把进程归属到对应GPU
        uuid_to_index = {}
        for line in uuid_out.strip().splitlines():
            if not line.strip():
                continue
            idx, uuid = [x.strip() for x in line.split(",", 1)]
            uuid_to_index[uuid] = idx

        gpus = []
        for line in gpu_out.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            index, name, mem_used, mem_total, util, temp = parts
            gpus.append({
                "index": int(index),
                "name": name,
                "mem_used_mb": int(float(mem_used)),
                "mem_total_mb": int(float(mem_total)),
                "util_pct": int(float(util)),
                "temp_c": int(float(temp)),
                "processes": [],
            })
        gpu_by_index = {g["index"]: g for g in gpus}

        for line in proc_out.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            gpu_uuid, pid, pname, used_mem = parts
            idx = uuid_to_index.get(gpu_uuid)
            if idx is not None and int(idx) in gpu_by_index:
                gpu_by_index[int(idx)]["processes"].append({
                    "pid": pid,
                    "name": pname,
                    "mem_mb": int(float(used_mem)) if used_mem not in ("[N/A]", "") else None,
                })

        return gpus

    def tail_log(self, n: int = 100) -> str:
        log_path = self.cfg["log_path"]
        try:
            _, out, _ = self._exec(f"tail -n {n} {log_path}")
            return out
        except Exception as e:
            return f"[读取日志失败: {e}]"

    async def health_check(self) -> bool:
        """探测 vLLM 的 /v1/models 接口是否已就绪(健康检查走HTTP,不走SSH)"""
        url = f"http://{self.cfg['host']}:{self.cfg['vllm_port']}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(url)
                return r.status_code == 200
        except Exception:
            return False
