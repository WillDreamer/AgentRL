import asyncio
import logging
import random
from typing import List, Tuple
import aiohttp
import os


async def _post_snippet(endpoint: str, payload: dict, *, client_timeout: float = 30.0) -> dict:
    """Low-level HTTP POST to the sandbox and return parsed JSON."""
    timeout = aiohttp.ClientTimeout(total=client_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(endpoint, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def single_sandbox(
    code: str,
    stdin: str = "",
    endpoint: str = "",
    language: str = "python",
    compile_timeout: float = 1.0,
    run_timeout: float = 3.0,
    semaphore: asyncio.Semaphore | None = None,
    max_attempts: int = 2,
) -> dict:
    """Submit *one* code snippet to the sandbox.

    Parameters
    ----------
    code : str
        Source code to execute.
    language : str, default ``"python"``
        Language tag supported by the sandbox.
    compile_timeout / run_timeout : float
        Same semantics as the HTTP API.
    semaphore : :class:`asyncio.Semaphore`, optional
        Concurrency guard supplied by the caller.  When *None*, a dummy
        semaphore of weight 1 is created so the coroutine can run
        standalone.
    max_attempts : int, default 2
        Number of retry attempts on network errors.

    Returns
    -------
    dict
        Parsed JSON from the sandbox.  In case of repeated failures, a
        minimal ``{"status": "Error", "error": str(exc)}`` dict is
        returned so the signature stays stable.
    """

    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    payload = {
        "code": code,
        "stdin": stdin if stdin is not None else "",
        "language": language,
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
    }

    for attempt in range(1, max_attempts + 1):
        try:
            async with semaphore:
                response = await _post_snippet(endpoint, payload)
            break  # Success ⇒ break retry loop
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "single_sandbox attempt %s/%s failed: %s", attempt, max_attempts, exc
            )
            if attempt == max_attempts:
                response = {"status": "runtime_error", "error": str(exc)}
            else:
                # Exponential backoff with jitter
                await asyncio.sleep(0.5 * 2 ** (attempt - 1) + random.random() * 0.1)
                continue

    # Mimic the original helper’s small pause, useful for rate limiting.
    await asyncio.sleep(2)
    return response


async def parallel_sandbox(
    tasks: List[str],
    stdin_list: List[str] = None,
    num_processes: int = 200,
) -> Tuple[List[bool], List[str], List[str]]:
    """Execute multiple snippets concurrently and aggregate results.

    Returns three lists of equal length ``len(tasks)``:

    * ``ok_flags``: *True* if the snippet ran with status == ``success``.
    * ``stdouts``: *stdout* text collected from the sandbox.
    * ``stderrs``: *stderr* text collected from the sandbox.
    """
    # endpoint = os.getenv("SANDBOX_ENDPOINT", None)
    endpoint = 'http://127.0.0.1:12345/faas/sandbox/'
    assert endpoint is not None, "SANDBOX_ENDPOINT is not set"
    semaphore = asyncio.Semaphore(num_processes)
    if stdin_list is None:
        tasks = [single_sandbox(code=code, endpoint=endpoint, semaphore=semaphore) for code in tasks]
    else:
        assert len(tasks) == len(stdin_list), f"len(tasks) ({len(tasks)}) != len(stdin_list) ({len(stdin_list)})"
        tasks = [single_sandbox(code=code, stdin=stdin, endpoint=endpoint, semaphore=semaphore) for code, stdin in zip(tasks, stdin_list)]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    ok_flags: List[bool] = []
    stdouts: List[str] = []
    stderrs: List[str] = []

    for r in results:
        ok_flags.append(r.get("status") == "success")
        run_res = r.get("run_result", {}) if isinstance(r, dict) else {}
        stdouts.append(run_res.get("stdout", ""))
        stderrs.append(run_res.get("stderr", ""))

    return ok_flags, stdouts, stderrs


if __name__ == "__main__":
    code_list = ["""
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
""", """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b, v
    return a

print([fib(i) for i in range(10)])
""", """
name = input("Your name:")
print(f"Hi, {name}!")
"""]
    stdin_list = ["", "", "Alice"]

    for code, stdin in zip(code_list, stdin_list):
        print(f"code: {code}")
        sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(parallel_sandbox(tasks=[code], stdin_list=[stdin]))

        print(f"sandbox_success: {sandbox_success}")
        print(f"sandbox_stdout: {sandbox_stdout}")
        print(f"sandbox_stderr: {sandbox_stderr} {sandbox_stderr[0].splitlines()[-1:]}")
