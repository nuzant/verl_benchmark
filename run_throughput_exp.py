import os

batch_size = 128
rollout_n = 8

nnodes_range = [4]
model_size_range = ["1.5B"]
rollout_backend_range = ["sglang", "vllm"]
max_response_length_range = ["8192"]

VERL_SCRIPT_PATH = "./nips25/"
VERL_VLLM_BASE = "./nips25/vllm_base.sh"
VERL_SGLANG_BASE = "./nips25/sglang_base.sh"
SBATCH_SCRIPT_PATH = "./"
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
    max_prompt_length=512,
    ppo_mini_batch_size=None,
    ppo_max_token_len_per_gpu=32768,
    ulysses_sequence_parallel_size=1,
    rollout_tensor_model_parallel_size=1,
    rollout_gpu_memory_utilization=0.6
):
    if ppo_mini_batch_size is None:
        ppo_mini_batch_size = batch_size
    
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


nnodes_range = [4]
model_size_range = ["1.5B"]
rollout_backend_range = ["sglang", "vllm"]
max_response_length_range = ["8192"]

generate_verl_script(
    batch_size=batch_size,
    rollout_n=rollout_n,
    nnodes=nnodes_range[0],
    model_size=model_size_range[0],
    rollout_backend="sglang",
    max_response_length=max_response_length_range[0],
)