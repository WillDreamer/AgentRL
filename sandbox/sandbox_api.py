"""
Minimal Firejail-based sandbox HTTP API.
Dependencies:
    pip install "fastapi[all]" uvicorn
System prerequisites:
    * firejail installed (sudo apt install firejail)
    * a non-login user called `sandboxer` (sudo useradd -r -M -s /usr/sbin/nologin sandboxer)
    * a Firejail profile at /etc/firejail/sandbox.profile such as:
        # /etc/firejail/sandbox.profile
        env none                    # start with a clean env
        env keep PATH
        env keep LANG
        env keep PYTHONIOENCODING
        net none                    # disable networking
        private                     # private filesystem rooted at --private dir
        rlimit as 512M              # memory cap
        rlimit cpu 3                # cpu-time cap
        rlimit nproc 50             # process count cap
        caps.drop all               # drop all capabilities
        seccomp
        whitelist /usr/bin/python3
        include /etc/firejail/whitelist-common.inc

Run the service with:
    uvicorn sandbox_api:app --host 0.0.0.0 --port 8000 --workers 4

The API is compatible with the `RunCodeRequest` / `RunResult` schema used by
ByteIntl Seed-Sandbox. A simple curl test:
    curl -X POST http://127.0.0.1:8000/faas/sandbox/ \
         -H 'Content-Type: application/json' \
         -d '{"code":"print(2+2)","language":"python","compile_timeout":1,"run_timeout":3}'
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Pydantic models ----------------

class RunStatus(str, Enum):
    """Execution outcome."""
    success = "success"
    timeout = "timeout"
    runtime_error = "runtime_error"


class RunCodeRequest(BaseModel):
    """Incoming JSON body from the client."""

    code: str
    stdin: str = ""
    language: str = "python"
    compile_timeout: float = 1.0  # kept for sdk compatibility, unused here
    run_timeout: float = 3.0


class RunResult(BaseModel):
    """JSON response back to the client."""

    status: RunStatus
    run_result: dict
    created_at: datetime


# ---------------- Core runner ----------------

async def _run_in_firejail(code: str, timeout: float, stdin_data: str = "") -> dict:
    """Execute *code* inside a fresh Firejail sandbox and return stdout/stderr."""

    # 1) Write user code to a tmpfs directory → zero-copy, fast cleanup
    workdir = Path(tempfile.mkdtemp(prefix="fj_", dir="/dev/shm"))
    src = workdir / "main.py"
    src.write_text(code)

    # 2) Build Firejail command line
    cmd = [
        "firejail",
        "--quiet",
        "--profile=/etc/firejail/sandbox.profile",
        f"--private={workdir}",
        "--net=none",              # disable network
        "--",
        "python3",
        src.name,
    ]

    # 3) Strip environment to stay below Firejail's MAX_ENVS=256 limit
    whitelist = ("PATH", "LANG", "LC_ALL", "PYTHONIOENCODING", "TERM")
    clean_env = {k: os.environ[k] for k in whitelist if k in os.environ}

    # 4) Launch subprocess under asyncio, enforce wall-clock timeout
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
        env=clean_env,
    )

    try:
        input_bytes = (stdin_data + "\n").encode() if len(stdin_data) > 0 else None
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_bytes),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        shutil.rmtree(workdir, ignore_errors=True)
        return {
            "status": RunStatus.timeout,
            "stdout": "",
            "stderr": "Timeout\n",
        }

    status = RunStatus.success if proc.returncode == 0 else RunStatus.runtime_error

    # 5) Clean up tmpfs directory
    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "status": status,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
    }


# ---------------- FastAPI wiring ----------------

app = FastAPI()
POOL = asyncio.Semaphore(200)  # gate per-process concurrency; tune to your CPU


@app.post("/faas/sandbox/", response_model=RunResult)
async def run_code(req: RunCodeRequest):
    """HTTP endpoint: compatible with the Seed-Sandbox client SDK."""

    if req.language != "python":
        raise HTTPException(400, "Only Python is supported in this minimal demo.")

    async with POOL:
        result = await _run_in_firejail(req.code, req.run_timeout, req.stdin)

    return RunResult(status=result["status"], run_result=result, created_at=datetime.utcnow())
