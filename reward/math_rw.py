import json
import os
import signal
import subprocess
import uuid
from typing import *

def compute_math_rw(data_source, solution_str, ground_truth, extra_info=None):
    tmp_id = str(uuid.uuid4())
    # print(tmp_id)
    # print(solution_str)
    with open(f"/tmp/{tmp_id}-input.jsonl", "w", encoding="utf-8") as f:
        for cur_solution in ground_truth:
            f.write(json.dumps({"answer": solution_str, "solution": cur_solution}) + "\n")

    venv_python = "/sympy/bin/python3"
    # print(f"math verify working dir: `{os.getcwd()}`")
    pro = subprocess.Popen(
        " ".join(
            [
                venv_python,
                "math_verify_utils_qwen.py",
                "--tmp_id",
                tmp_id,
            ]
        ),
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
    )
    pro.wait()
    try:
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    label = 0
    try:
        with open(f"/tmp/{tmp_id}-output.jsonl", "r") as f:
            for line in f.readlines():
                output_data = json.loads(line)
                label = output_data["retval"] or label
    except FileNotFoundError as e:
        # The subprocess may fail to parse the input (maybe due to reaching the maximum recursion length)
        # We just return 0 for the reward.
        print(
            f"Failed to parse: solution_str `{solution_str[-1000:]}` ground_truth `{ground_truth}`. Set 0 reward."
        )
        label = 0
    finally:
        # if os.path.exists(f"/tmp/{tmp_id}-input.jsonl"):
        #     os.remove(f"/tmp/{tmp_id}-input.jsonl")
        # if os.path.exists(f"/tmp/{tmp_id}-output.jsonl"):
        #     os.remove(f"/tmp/{tmp_id}-output.jsonl")
        print(f" >>> Reward = {label}")
        return label
    # print(f"in math rw, {solution_str} {ground_truth}")
    # import time
    # time.sleep(0.1)
    # return 1


if __name__ == "__main__":
    sample = {
        "answers": ["-\\frac{2}{3}"],
        "solutions": [
            "1. **Apply the operation $\\otimes$ to the innermost parentheses first:**\n   \\[\n   (1 \\otimes 2) \\otimes 3 = \\left(\\frac{1^2}{2}\\right) \\otimes 3 = \\frac{1}{2} \\otimes 3\n   \\]\n   \\[\n   1 \\otimes (2 \\otimes 3) = 1 \\otimes \\left(\\frac{2^2}{3}\\right) = 1 \\otimes \\frac{4}{3}\n   \\]\n\n2. **Calculate each part using the definition of $\\otimes$:**\n   \\[\n   \\frac{1}{2} \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n   \\]\n   \\[\n   1 \\otimes \\frac{4}{3} = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n   \\]\n\n3. **Subtract the two results:**\n   \\[\n   \\left(\\frac{1}{12}\\right) - \\left(\\frac{3}{4}\\right) = \\frac{1}{12} - \\frac{9}{12} = -\\frac{8}{12} = -\\frac{2}{3}\n   \\]\n\n4. **Conclude with the final answer:**\n   \\[\n   \\boxed{A}\n   \\]",
            "\\boxed{-\\frac{2}{3}}",
        ],
    }
    ans = "1. **Apply the operation $\\otimes$ to the innermost parentheses first:**\n   \\[\n   (1 \\otimes 2) \\otimes 3 = \\left(\\frac{1^2}{2}\\right) \\otimes 3 = \\frac{1}{2} \\otimes 3\n   \\]\n   \\[\n   1 \\otimes (2 \\otimes 3) = 1 \\otimes \\left(\\frac{2^2}{3}\\right) = 1 \\otimes \\frac{4}{3}\n   \\]\n\n2. **Calculate each part using the definition of $\\otimes$:**\n   \\[\n   \\frac{1}{2} \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n   \\]\n   \\[\n   1 \\otimes \\frac{4}{3} = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n   \\]\n\n3. **Subtract the two results:**\n   \\[\n   \\left(\\frac{1}{12}\\right) - \\left(\\frac{3}{4}\\right) = \\frac{1}{12} - \\frac{9}{12} = -\\frac{8}{12} = -\\frac{2}{3}\n   \\]\n\n4. **Conclude with the final answer:**\n   \\[\n   \\boxed{-\\frac{2}{3}}\n   \\"
    compute_math_rw("", sample["solutions"][0], ["\\boxed{-\\frac{2}{3}}"])