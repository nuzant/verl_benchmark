import json
import re
import os

def parse_data_sglang(fn):
    base = os.path.basename(fn)
    pattern = r'^(?P<jobid>[^-]+)-verl-sglang-(?P<model_size>\d+\.?\d*)B-n(?P<n_nodes>\d+)-ctx(?P<ctx>\d+)-(?P<bs>\d+)x(?P<n>\d+)\-2025051400.out$'
    match = re.fullmatch(pattern, base)

    if not match:
        return 

    res = match.groupdict()
    
    with open(fn, "r") as f:
        lines = f.readlines()
        time_per_step = []
        gen_tokens_per_step = []
        for line in lines:
            if "out of memory" in line:
                time_per_step = ["OOM"]
                res["times"] = time_per_step
                return None
            if "perf/time_per_step" not in line and "response_length/mean" not in line:
                continue
            words = line.split(" ")
            for word in words:
                if "perf/time_per_step" in word:
                    t = float(word.split(":")[1])
                    time_per_step.append(t)
                if "response_length/mean" in word:
                    t = float(word.split(":")[1])
                    gen_tokens_per_step.append(t)
    if len(time_per_step) != 4:
        return
    res["times"] = time_per_step
    res["avg_resp_len"] = gen_tokens_per_step
    if "OOM" not in time_per_step:
        res["effective_throughput"] = sum(gen_tokens_per_step) * 512 * 16 / sum(time_per_step)
    return res


def parse_data_vllm(fn):
    base = os.path.basename(fn)
    pattern = r'^(?P<jobid>[^-]+)-verl-vllm-(?P<model_size>\d+\.?\d*)B-n(?P<n_nodes>\d+)-ctx(?P<ctx>\d+)-(?P<bs>\d+)x(?P<n>\d+)\.out$'
    match = re.fullmatch(pattern, base)

    if not match:
        return 

    res = match.groupdict()
    if res["jobid"] == "2923":
        return
    
    with open(fn, "r") as f:
        lines = f.readlines()
        time_per_step = []
        gen_tokens_per_step = []
        for line in lines:
            if "out of memory" in line:
                time_per_step = ["OOM"]
                res["times"] = time_per_step
                return None
            if "perf/time_per_step" not in line and "response_length/mean" not in line:
                continue
            words = line.split(" ")
            for word in words:
                if "perf/time_per_step" in word:
                    t = float(word.split(":")[1])
                    time_per_step.append(t)
                if "response_length/mean" in word:
                    t = float(word.split(":")[1])
                    gen_tokens_per_step.append(t)
    if len(time_per_step) < 3:
        return
    res["times"] = time_per_step
    res["avg_resp_len"] = gen_tokens_per_step
    if "OOM" not in time_per_step:
        res["effective_throughput"] = sum(gen_tokens_per_step) * 512 * 16 / sum(time_per_step)
    return res


if __name__ == "__main__":
    log_path = "/storage/openpsi/users/meizhiyu.mzy/nips25/logs"
    output_path = "/storage/openpsi/users/meizhiyu.mzy/nips25/verl_tp.jsonl"
    r = []
    for fn in os.listdir(log_path):
        abs_fn = os.path.join(log_path, fn)
        res = parse_data_sglang(abs_fn)
        if res is not None:
            if "times" in res and res["times"]:
                # print(res)
                res["backend"] = "sglang"
                r.append(res)
    
    for fn in os.listdir(log_path):
        abs_fn = os.path.join(log_path, fn)
        res = parse_data_vllm(abs_fn)
        if res is not None:
            if "times" in res and res["times"]:
                # print(res)
                res["backend"] = "vllm"
                r.append(res)
    
    r.sort(key = lambda x: (float(x["model_size"]), int(x["ctx"]), int(x["n_nodes"]), int(x["jobid"])))
    for res in r:
        print(res)
    with open(output_path, "w") as f:
        for res in r:
            line = json.dumps(res)
            f.write(f"{line}\n")
