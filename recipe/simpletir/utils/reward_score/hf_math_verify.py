# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from itertools import product

from math_verify import parse
from math_verify.grader import sympy_expr_eq
from sympy import Basic, MatrixBase

from recipe.simpletir.utils.reward_score.qwen_math_eval_toolkit.parser import (
    extract_answer as qwen_extract_answer,
)


def extract_last_boxed(text):
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"

    matches = list(re.finditer(pattern, text))

    if matches:
        return matches[-1].group(0)
    return None


def extract_solution(solution_str):
    model_output = re.sub(
        r"^.*?<\|im_start\|>assistant",
        "<|im_start|>assistant",
        solution_str,
        flags=re.DOTALL,
        count=1,
    )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "[END]"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False


def verify_without_timeout(
    gold,
    target,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
) -> bool:
    from math_verify.utils import timeout

    @timeout(5)
    def compare_single_extraction(
        gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str
    ) -> bool:
        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(
            target, (Basic, MatrixBase)
        ):
            return sympy_expr_eq(
                gold, target, float_rounding, numeric_precision, strict
            )
        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        return False

    def compare_single_extraction_wrapper(g, t):
        try:
            return compare_single_extraction(g, t)
        except Exception as e:
            return False

    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(target, list):
        target = [target]

    return any(
        compare_single_extraction_wrapper(g, t) for g, t in product(gold, target)
    )


def hf_verify_with_try(gold, target):
    try:
        parsed_target = parse(target)
        parsed_gold = parse(gold)
        # we have removed the timeout to make it work in async
        return verify_without_timeout(gold=parsed_gold, target=parsed_target)
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


def compute_score(solution_str, ground_truth):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    if "\\boxed" not in extract_answer:
        boxed_answer = f"\\boxed{{{extract_answer}}}"
    else:
        boxed_answer = extract_answer

    if "\\boxed" not in ground_truth:
        boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
    else:
        boxed_ground_truth = ground_truth

    correct = hf_verify_with_try(gold=boxed_ground_truth, target=boxed_answer)

    if correct:
        answer_accuracy = 1
    else:
        answer_accuracy = 0

    total_score = answer_accuracy

    if is_boxed_matched:
        format_reward = 1
    else:
        format_reward = 0

    has_code_piece = re.search(
        r"```(?:py|python)?\n(.*?)\n```\nCode execution result:",
        solution_str,
        re.DOTALL,
    )

    # we include score here for calculating its varaince in extra_info
    return {
        "score": total_score,
        "extra_info": {
            "is_boxed_ratio": format_reward,
            "score": total_score,
            "valid_code": 1 if has_code_piece else 0,
        },
    }
