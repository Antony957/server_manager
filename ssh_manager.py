"""
ssh_manager.py
通过 SSH 在远程服务器上管理 vLLM 的生命周期、配置下发及集群探活 (纯端口驱动·完全废弃PID文件版)
"""
import shlex
import json
import paramiko
import httpx


class SSHManager:
    def __init__(self, server_cfg: dict):
        """
        server_cfg 结构来自于 servers.yaml，例如:
        {
            "id": "srv1", "name": "...", "host": "10.200.14.160",
            "log_path": "/tmp/vllm.log", "models": [...]
        }
        """
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

    def _get_runtime_port(self) -> int:
        """从内存数据结构中自适应抓取当前模型持有的端口"""
        if self.cfg.get("models") and len(self.cfg["models"]) > 0:
            return self.cfg["models"][0]["vllm_config"].get("port", 33261)
        return 33261

    def is_process_alive(self) -> bool:
        """根据绑定的端口号反查进程是否存在 (兼容性最优版)"""
        port = self._get_runtime_port()
        # 查找监听该端口的套接字。在行尾加空格做精准匹配，防止多位端口误抓
        cmd = f"ss -tlnp | grep -E ':{port} ' && echo ALIVE || echo DEAD"
        try:
            _, out, _ = self._exec(cmd)
            return "ALIVE" in out
        except Exception:
            return False

    def start_vllm(self, model_cfg: dict, gpus: list[int] | None = None) -> dict:
        """
        基于官方 --config 托管文件进行优雅启动 (正宗 YAML 落地版)
        """
        import yaml  # 确保函数内或文件头引入了 yaml
        log_path = self.cfg["log_path"]

        # 1. 解构出核心参数并把并行参数合入 vllm 官方规范字典
        tp = model_cfg.get("tensor_parallel_size", 1)
        pp = model_cfg.get("pipeline_parallel_size", 1)

        vllm_args = model_cfg["vllm_config"].copy()
        vllm_args["tensor_parallel_size"] = tp
        vllm_args["pipeline_parallel_size"] = pp

        # 2. 先强行对当前端口执行一次优雅收尸
        self.stop_vllm(ignore_errors=True)

        # 3. 🎯 【终极修正】：用 yaml.safe_dump 生成纯正的、带标准缩进的 YAML 格式字符串！
        # default_flow_style=False 确保输出的是标准的“换行+缩进”格式，而不是单行大括号格式
        config_yaml_str = yaml.safe_dump(vllm_args, default_flow_style=False, allow_unicode=True)

        # 打印一下看看，这回绝对是标准干净的 YAML 了
        print("--- 生成的标准 vLLM 配置文件 ---")
        print(config_yaml_str)

        remote_config_path = f"/tmp/vllm_runtime_config_{vllm_args.get('port', 33261)}.yaml"

        # 依然通过 Linux EOF 覆写过去，因为没有任何大括号纠缠，解析极其安全
        write_config_cmd = f"cat << 'EOF' > {remote_config_path}\n{config_yaml_str}\nEOF"
        self._exec(write_config_cmd)

        # 4. 卡号环境变量拼装
        env_prefix = ""
        if gpus and len(gpus) > 0:
            env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in gpus)} "

        # 5. 指向托管配置文件的超精简启动命令
        cmd = (
            "source $(conda info --base)/etc/profile.d/conda.sh && "
            "conda activate llmserver && "
            f"( {env_prefix}vllm serve --config {remote_config_path} "
            f"> {log_path} 2>&1 < /dev/null & )"
        )
        full_cmd = f"bash -lc {shlex.quote('source ~/.bashrc 2>/dev/null; ' + cmd)}"
        exit_code, out, err = self._exec(full_cmd)
        return {"exit_code": exit_code, "stdout": out, "stderr": err}


    def stop_vllm(self, ignore_errors: bool = False) -> dict:
        """先礼后兵两阶段优雅退出法：优先让 vLLM 释放显存，最后强杀扫尾 (干掉了 rm pidfile)"""
        port = self._get_runtime_port()

        # 1. 抓取端口对应的 PID -> 2. 发送 SIGTERM(kill) 优雅回收 CUDA 上下文 -> 3. 稳妥等3秒 -> 4. 强杀兜底
        cmd = (
            f"PID=$(ss -tlnp | grep -E ':{port} ' | grep -E 'python|vllm' | awk '{{print $NF}}' | cut -d, -f2 | cut -d= -f2); "
            f"if [ ! -z \"$PID\" ]; then "
            f"  if [ \"$(ps -o user= -p $PID 2>/dev/null)\" = \"chongwen\" ]; then "
            f"    kill $PID 2>/dev/null; "
            f"    sleep 3; "
            f"    kill -9 $PID 2>/dev/null; "
            f"  fi "
            f"fi"
        )
        try:
            exit_code, out, err = self._exec(f"bash -lc {shlex.quote('source ~/.bashrc 2>/dev/null; ' + cmd)}")
            return {"exit_code": exit_code, "stdout": out, "stderr": err}
        except Exception as e:
            if ignore_errors:
                return {"exit_code": -1, "stdout": "", "stderr": str(e)}
            raise

    # ---------------------------------------------------------------------------
    # 通用任务执行 (SSH + 命令行)
    # ---------------------------------------------------------------------------
    def run_task(self, command: str, log_path: str, gpus: list[int] | None = None) -> dict:
        """
        在远程服务器后台执行任意命令行任务，输出重定向到 log_path，并返回后台进程 PID。
        用于「任务管理」模块：用户自由输入命令，经 SSH 在目标服务器拉起。
        gpus: 可选，指定要使用的 GPU 编号，会在命令前注入 CUDA_VISIBLE_DEVICES 环境变量。
        """
        # 1. 先清空日志文件
        self._exec(f"> {log_path}")

        # 2. 卡号环境变量拼装（放在命令最前面）
        env_prefix = ""
        if gpus and len(gpus) > 0:
            env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in gpus)} "

        # 3. 用 nohup 后台运行，并通过 echo $! 捕获 PID
        wrapped = f"source ~/.bashrc 2>/dev/null; nohup {env_prefix}{command} > {log_path} 2>&1 < /dev/null & echo $!"
        full_cmd = f"bash -lc {shlex.quote(wrapped)}"
        exit_code, out, err = self._exec(full_cmd)

        # 取输出最后一行作为 PID
        pid = ""
        if out.strip():
            pid = out.strip().splitlines()[-1].strip()
        return {"exit_code": exit_code, "pid": pid, "stdout": out, "stderr": err}

    def stop_task(self, pid: str) -> dict:
        """根据 PID 强杀任务进程及其子进程 (用于「任务管理」停止功能)"""
        if not pid:
            return {"exit_code": -1, "stdout": "", "stderr": "无可用 PID"}
        cmd = (
            f"kill -9 -{pid} 2>/dev/null; "      # 杀掉整个进程组
            f"kill -9 {pid} 2>/dev/null; "        # 兜底直杀
            f"pkill -P {pid} 2>/dev/null; "       # 杀子进程
            f"echo done"
        )
        try:
            exit_code, out, err = self._exec(f"bash -lc {shlex.quote('source ~/.bashrc 2>/dev/null; ' + cmd)}")
            return {"exit_code": exit_code, "stdout": out, "stderr": err}
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

    def get_gpu_status(self) -> list[dict]:
        """用 nvidia-smi 查询每张卡的状态"""
        combined = (
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
            " && echo '---GPU_PROC---' && "
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits"
            " && echo '---GPU_UUID---' && "
            "nvidia-smi --query-gpu=index,uuid --format=csv,noheader"
        )
        try:
            _, out, _ = self._exec(combined)
            if "---GPU_PROC---" not in out or "---GPU_UUID---" not in out:
                return []
        except Exception:
            return []

        gpu_part, rest = out.split("---GPU_PROC---\n", 1)
        proc_part, uuid_part = rest.split("---GPU_UUID---\n", 1)
        gpu_out = gpu_part.strip()
        proc_out = proc_part.strip()
        uuid_out = uuid_part.strip()

        uuid_to_index = {}
        for line in uuid_out.strip().splitlines():
            if not line.strip(): continue
            parts = line.split(",")
            if len(parts) >= 2:
                idx, uuid = parts[0].strip(), parts[1].strip()
                uuid_to_index[uuid] = idx

        gpus = []
        for line in gpu_out.strip().splitlines():
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6: continue
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
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4: continue
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

    # ---------------------------------------------------------------------------
    # 任务进程存活探测 + 远程文件下载
    # ---------------------------------------------------------------------------
    def is_process_alive(self, pid: str) -> bool:
        """根据 PID 判断远程任务进程是否仍在运行"""
        if not pid:
            return False
        cmd = f"kill -0 {pid} 2>/dev/null && echo ALIVE || echo DEAD"
        try:
            _, out, _ = self._exec(cmd)
            return "ALIVE" in out
        except Exception:
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """用 paramiko SFTP 把远程文件下载到本机"""
        import os
        client = self._connect()
        try:
            sftp = client.open_sftp()
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            sftp.get(remote_path, local_path)
            sftp.close()
            return True
        except Exception as e:
            print(f"[下载失败] {remote_path} -> {local_path}: {e}")
            return False
        finally:
            client.close()
