# verl Benchmark

# Versions and Images

This benchmark is based on verl main branch, commit 76084d36cbca4906539a6071d8e12a1429e94698 on May 7 2025. 

Docker images:
- vLLM v0.8.5: `whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3`
- SGLang v0.4.6.post5: `ocss884/verl-sglang:ngc-th2.6.0-cu126-sglang0.4.6.post5`

# Generating scripts

To generate verl scripts (in ./nips25/) and sbatch scripts for slurm (in ./sbatch_scripts/), you could change the model, image, and working directory to paths in your own filesystem in `./nips25/[sglang/vllm]_base.sh` and `./sbatch_run_ray_base_[sglang/vllm].sh`, then run `python3 run_throughput_exp.py`.

# Run benchmark directly with Ray cluster

If you have already setup a ray cluster based on verl docker image, you could directly run bash scripts generated in `./nips25/` folder to run the benchmark.

# Run benchmark with slurm + ray

If you have a slurm cluster, you could automatically run all generated experiments as slurm jobs with `python3 run_throughput_exp.py`. You could also run a single benchmark experiment, by executing `sbatch sbatch_scripts/sbatch-xxxx.sh`. These sbatch scripts will automatically setup ray cluster on slurm nodes, run benchmark scripts, and clean up slurm jobs. 