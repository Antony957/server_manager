"""
task_db.py
任务管理的本地 SQLite 存储层
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = "tasks.db"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id               TEXT PRIMARY KEY,
        name             TEXT,
        server_id        TEXT,
        server_name      TEXT,
        command          TEXT,
        pid              TEXT,
        status           TEXT,   -- pending / running / stopped / finished / error / vllm_starting
        remote_log_path  TEXT,
        local_log_path   TEXT,
        created_at       TEXT,
        started_at       TEXT,
        finished_at      TEXT,
        is_vllm          INTEGER DEFAULT 0
    )
    """)
    # 兼容旧库：补齐新增列
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
    if "is_vllm" not in cols:
        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN is_vllm INTEGER DEFAULT 0")
        except Exception:
            pass
    conn.commit()
    conn.close()


def create_task(task_id: str, name: str, server_id: str, server_name: str,
                command: str, remote_log_path: str, is_vllm: bool = False) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """INSERT INTO tasks
           (id, name, server_id, server_name, command, pid, status, remote_log_path, local_log_path, created_at, is_vllm)
           VALUES (?, ?, ?, ?, ?, '', 'pending', ?, '', ?, ?)""",
        (task_id, name, server_id, server_name, command, remote_log_path, _now(), int(bool(is_vllm))),
    )
    conn.commit()
    row = cur.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return dict(row)


def get_task(task_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_tasks():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_task(task_id: str, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [task_id]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"UPDATE tasks SET {cols} WHERE id=?", vals)
    conn.commit()
    conn.close()


def next_seq() -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()
    conn.close()
    return (row[0] if row else 0) + 1