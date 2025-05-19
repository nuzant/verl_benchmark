import os
import itertools
import argparse
import subprocess

batch_size = 768
rollout_n = 16

nnodes_range = [48]
# model_size_range = ["1.5B", "7B", "14B", "32B"]
model_size_range = ["32B"]
# rollout_backend_range = ["vllm", "sglang"]
rollout_backend_range = ["vllm"]
max_response_length_range = [8192, 16384, 32768]

VERL_SCRIPT_PATH = "./nips25/"
VERL_VLLM_BASE = "./nips25/vllm_base.sh"
VERL_SGLANG_BASE = "./nips25/sglang_base.sh"
SBATCH_SCRIPT_PATH = "./sbatch_scripts/"
SBATCH_VLLM_BASE = "./sbatch_run_ray_base_vllm.sh"
SBATCH_SGLANG_BASE = "./sbatch_run_ray_base_sglang.sh"

verl_script_name = "run-{rollout_backend}-{model_size}-n{nnodes}-ctx{max_response_length}-{batch_size}x{rollout_n}.sh"
sbatch_script_name = "sbatch-{rollout_backend}-{model_size}-n{nnodes}-ctx{max_response_length}-{batch_size}x{rollout_n}.sh"

def generate_verl_script(
    batch_size,
    rollout_n,
    nnodes,
    model_size,
    rollout_backend,
    max_response_length,
    max_prompt_length=1024,
    ppo_mini_batch_size=None,
    ppo_max_token_len_per_gpu=32768,
    ulysses_sequence_parallel_size=1,
    rollout_tensor_model_parallel_size=1,
    rollout_gpu_memory_utilization=0.6
):
    print(f"model size = {model_size}")
    if model_size == "32B":
        ppo_max_token_len_per_gpu = 24576
        rollout_tensor_model_parallel_size = 8
        rollout_gpu_memory_utilization = 0.4
    elif model_size == "7B":
        rollout_tensor_model_parallel_size = 1

    if ppo_mini_batch_size is None:
        ppo_mini_batch_size = batch_size//4
    if ppo_max_token_len_per_gpu < max_prompt_length + max_response_length:
        ppo_max_token_len_per_gpu = max_prompt_length + max_response_length

    if rollout_backend == "vllm":
        verl_base = VERL_VLLM_BASE
        sbatch_base = SBATCH_VLLM_BASE
    elif rollout_backend == "sglang":
        verl_base = VERL_SGLANG_BASE
        sbatch_base = SBATCH_SGLANG_BASE
    else:
        raise ValueError(f"rollout backend {rollout_backend} is not vllm or sglang")

    verl_script_path = os.path.join(
        VERL_SCRIPT_PATH, verl_script_name.format(
            rollout_backend=rollout_backend,
            model_size=model_size,
            nnodes=nnodes,
            max_response_length=max_response_length,
            batch_size=batch_size,
            rollout_n=rollout_n
        )
    )
    sbatch_script_path = os.path.join(
        SBATCH_SCRIPT_PATH, sbatch_script_name.format(
            rollout_backend=rollout_backend,
            model_size=model_size,
            nnodes=nnodes,
            max_response_length=max_response_length,
            batch_size=batch_size,
            rollout_n=rollout_n
        )
    )
    
    with open(verl_base, "r") as f:
        verl_script_content = f.read()
    verl_script_content = verl_script_content.format(
        batch_size=batch_size,
        rollout_n=rollout_n,
        nnodes=nnodes,
        model_size=model_size,
        rollout_backend=rollout_backend,
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_max_token_len_per_gpu=ppo_max_token_len_per_gpu,
        ulysses_sequence_parallel_size=ulysses_sequence_parallel_size,
        rollout_tensor_model_parallel_size=rollout_tensor_model_parallel_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization
    )

    with open(sbatch_base, "r") as f:
        sbatch_script_content = f.read()
    print(sbatch_script_content)
    sbatch_script_content = sbatch_script_content.format(
        batch_size=batch_size,
        rollout_n=rollout_n,
        nnodes=nnodes,
        model_size=model_size,
        rollout_backend=rollout_backend,
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_max_token_len_per_gpu=ppo_max_token_len_per_gpu,
        ulysses_sequence_parallel_size=ulysses_sequence_parallel_size,
        rollout_tensor_model_parallel_size=rollout_tensor_model_parallel_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization
    )

    with open(verl_script_path, "w") as f:
        print(f">>> Writing to {verl_script_path}:")
        print(verl_script_content)
        f.write(verl_script_content)
    print("\n")
    with open(sbatch_script_path, "w") as f:
        print(f">>> Writing to {sbatch_script_path}:")
        print(sbatch_script_content)
        f.write(sbatch_script_content)
    print("\n")
    return sbatch_script_path

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")

args = parser.parse_args()

for n_nodes, model_size, max_response_length in itertools.product(
    nnodes_range, model_size_range, max_response_length_range
):
    for rollout_backend in rollout_backend_range:
        path = generate_verl_script(
            batch_size=batch_size,
            rollout_n=rollout_n,
            nnodes=n_nodes,
            model_size=model_size,
            rollout_backend=rollout_backend,
            max_response_length=max_response_length,
            ppo_mini_batch_size=batch_size//4,
        )
        if not args.dry_run:
            print(f"running \'bash ./run_exp.sh {path}\'")
            r = subprocess.run(f"bash ./run_exp.sh {path}", shell=True)
            # Check the return code
            if r.returncode == 0:
                print(f"Command \'bash ./run_exp.sh {path}\' executed successfully:")
                print(r.stdout)
            else:
                print(f"Command \'bash ./run_exp.sh {path}\' failed with error code: {r.returncode}")
                print(r.stderr)