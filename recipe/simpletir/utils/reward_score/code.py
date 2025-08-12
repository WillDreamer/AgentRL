import asyncio
import json
import os
import time

import numpy as np

if os.getenv("SANDBOX_ENDPOINT", None) is not None:
    from sandbox.local_sandbox import parallel_sandbox
else:
    from sandbox.internal_sandbox import parallel_sandbox

from recipe.simpletir.agent_utils import truncate_content

MAX_CHAR_DISPLAY = 2048


def compute_score(solution_str, ground_truth, extra_info):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()

    reward_log = []

    has_code_piece = len(solution_str) != 0

    if not has_code_piece:
        reward_log.append("-" * 16 + "No Code Detected!" + "-" * 16)
        reward_log.append("-" * 16 + "Original Model Output" + "-" * 16)
        reward_log.append(solution_str)
        reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
        # reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))

        reward_log = "\n".join(reward_log)
        reward_log = (
            "❌" * 16
            + "Reward Calculation"
            + "❌" * 16
            + "\n"
            + reward_log
            + "\n"
            + "❌" * 16
            + f"Final Reward = {0.0}"
            + "❌" * 16
        )
        # print(reward_log + "\n\n")
        return {
            "score": 0.0,
            "extra_info": {
                "score": 0.0,
                "valid_code": 1 if has_code_piece else 0,
            },
        }

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    ground_truth = json.loads(ground_truth)

    t_start = time.time()

    if isinstance(ground_truth, list):  # livecodebench
        stdin_list = []
        stdout_list = []
        code_list = []
        for test in ground_truth:
            if test["testtype"] == "stdin":
                stdin = test["input"]
                stdin_list.append(stdin)
                stdout = test["output"]
                stdout_list.append(stdout)
                code_list.append(solution_str)
            elif test["testtype"] == "functional":
                exec_lines = [solution_str]
                exec_lines.append("sol = Solution()")

                func_name = test["metadata"]["func_name"]
                arg_lines = [
                    ln.strip()
                    for ln in test["input"].strip().split("\\n")
                    if ln.strip()
                ]
                args_str = ", ".join(arg_lines)
                exec_lines.append(f"print(sol.{func_name}({args_str}))")

                stdin = ""
                stdin_list.append(stdin)
                stdout = test["output"]
                stdout_list.append(stdout)
                code_to_execute = "\n".join(exec_lines)
                code_list.append(code_to_execute)

        reward_log.append(solution_str)

        sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(
            parallel_sandbox(code_list, stdin_list, num_processes=256)
        )

        for stdin, stdout, sandbox_stdout, sandbox_stderr in zip(
            stdin_list, stdout_list, sandbox_stdout, sandbox_stderr
        ):
            if len(sandbox_stderr) > 0 or sandbox_stdout.strip() != stdout.strip():
                reward_log.append(
                    "!" * 16
                    + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                    + "!" * 16
                )
                reward_log.append(f"🔎Input: {repr(stdin)}")
                reward_log.append(f"✅Expected: {repr(stdout.strip())}")
                if len(sandbox_stdout) > 0:
                    reward_log.append(
                        f"❌Actual stdout: {truncate_content(sandbox_stdout, max_length=512)}"
                    )
                if len(sandbox_stderr) > 0:
                    reward_log.append(
                        f"❌Actual stderr: {truncate_content(sandbox_stderr, max_length=512)}"
                    )
                reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                # reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))

                reward_log = "\n".join(reward_log)
                reward_log = (
                    "❌" * 16
                    + "Reward Calculation"
                    + "❌" * 16
                    + "\n"
                    + reward_log
                    + "\n"
                    + "❌" * 16
                    + f"Final Reward = {0.0}"
                    + "❌" * 16
                )
                # print(reward_log + "\n\n")
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "valid_code": 1 if has_code_piece else 0,
                    },
                }

    elif isinstance(ground_truth, dict):  # taco / LeetCode
        if "functional" in ground_truth:
            code_to_execute = solution_str + "\n" + ground_truth["functional"]
            reward_log.append(code_to_execute)
            sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(
                parallel_sandbox([code_to_execute], num_processes=256)
            )
            success = sandbox_success[0]
            stdout = str(sandbox_stdout[0])
            stderr = str(sandbox_stderr[0])

            if len(stderr) > 0:
                reward_log.append(
                    "!" * 16
                    + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                    + "!" * 16
                )
                reward_log.append(truncate_content(stdout, max_length=512))
                reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                # reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))

                reward_log = "\n".join(reward_log)
                reward_log = (
                    "❌" * 16
                    + "Reward Calculation"
                    + "❌" * 16
                    + "\n"
                    + reward_log
                    + "\n"
                    + "❌" * 16
                    + f"Final Reward = {0.0}"
                    + "❌" * 16
                )
                # print(reward_log + "\n\n")
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "valid_code": 1 if has_code_piece else 0,
                    },
                }
        elif "inputs" in ground_truth and "outputs" in ground_truth:
            reward_log.append(solution_str)
            stdin_list: str = ground_truth["inputs"]
            stdout_list: str = ground_truth["outputs"]

            sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(
                parallel_sandbox(
                    [solution_str] * len(stdin_list), stdin_list, num_processes=256
                )
            )

            for stdin, stdout, sandbox_stdout, sandbox_stderr in zip(
                stdin_list, stdout_list, sandbox_stdout, sandbox_stderr
            ):
                if len(sandbox_stderr) > 0 or sandbox_stdout.strip() != stdout.strip():
                    reward_log.append(
                        "!" * 16
                        + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                        + "!" * 16
                    )
                    reward_log.append(f"🔎Input: {repr(stdin)}")
                    reward_log.append(f"✅Expected: {repr(stdout.strip())}")
                    if len(sandbox_stdout) > 0:
                        reward_log.append(
                            f"❌Actual stdout: {truncate_content(sandbox_stdout, max_length=512)}"
                        )
                    if len(sandbox_stderr) > 0:
                        reward_log.append(
                            f"❌Actual stderr: {truncate_content(sandbox_stderr, max_length=512)}"
                        )
                    reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    # reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))

                    reward_log = "\n".join(reward_log)
                    reward_log = (
                        "❌" * 16
                        + "Reward Calculation"
                        + "❌" * 16
                        + "\n"
                        + reward_log
                        + "\n"
                        + "❌" * 16
                        + f"Final Reward = {0.0}"
                        + "❌" * 16
                    )
                    # print(reward_log + "\n\n")
                    return {
                        "score": 0.0,
                        "extra_info": {
                            "score": 0.0,
                            "valid_code": 1 if has_code_piece else 0,
                        },
                    }
        else:
            raise ValueError(
                f"ground truth is not functional or input/output: {ground_truth}"
            )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    if len(sandbox_stdout) > 0:
        reward_log.append(f"stdout: {truncate_content(sandbox_stdout, max_length=512)}")
    if len(sandbox_stderr) > 0:
        reward_log.append(f"stderr: {truncate_content(sandbox_stderr, max_length=512)}")
    reward_log = "\n".join(reward_log)
    reward_log = (
        "✅" * 16
        + "Reward Calculation"
        + "✅" * 16
        + "\n"
        + reward_log
        + "\n"
        + "✅" * 16
        + f"Final Reward = {1.0}"
        + "✅" * 16
    )
    # print(reward_log + "\n\n")
    return {
        "score": 1.0,
        "extra_info": {
            "score": 1.0,
            "valid_code": 1 if has_code_piece else 0,
        },
    }
