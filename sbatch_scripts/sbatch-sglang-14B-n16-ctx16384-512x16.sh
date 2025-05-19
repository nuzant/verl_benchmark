#!/bin/bash
#SBATCH --job-name=mzy-verl-sglang-14B-n16-ctx16384-512x16
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=10G
#SBATCH --exclude=slurmd-[20,40,63,78,88,94]
#SBATCH --output=/storage/openpsi/users/meizhiyu.mzy/nips25/logs/%j-verl-sglang-14B-n16-ctx16384-512x16-2025051400.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# load necessary modules

# replace these information with your own
workdir=/storage/openpsi/users/meizhiyu.mzy/run_verl/verl
apptainer_image_path=/storage/openpsi/images/verl-sglang.sif
# replace these information with your own

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo nodes=$nodes

nodes_array=($nodes)
echo node_array=$nodes_array

head_node=${nodes_array[0]}
echo head_node=$head_node

head_node_ip=$(srun --mpi=pmi2 --nodes=1 --ntasks=1 --nodelist="$head_node" hostname --ip-address)
echo head_node_ip=$head_node_ip

###### 

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

container_envs="--env CUDA_HOME=/usr/local/cuda \
    --env VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000 \
    --env LC_ALL=C \
    --env LANG=C \
    --env NCCL_SOCKET_IFNAME=bond0 \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-8726acfd20a41b5ab8ac35fd51014b15169fa20a \
    --env NCCL_NET_PLUGIN="" \
    --env NCCL_IB_GID_INDEX=3 \
    --env NCCL_IB_TIMEOUT=2 \
    --env NCCL_IB_RETRY_CNT=7 \
    --env NCCL_IB_SL=5 \
    --env NCCL_IB_TC=136 \
    --env NCCL_IB_HCA=mlx5_bond \
    --env NCCL_DEBUG=WARN \
    --env NCCL_IB_QPS_PER_CONNECTION=8 \
    --env NCCL_SET_THREAD_NAME=1 \
    --env NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH"
# printenv

echo "Starting HEAD at $head_node"
srun --mpi=pmi2 -K -l --chdir $workdir --nodes=1 --ntasks=1 \
    --gres=gpu:8 --cpus-per-task=120 --mem-per-cpu=8G -w "$head_node" \
    singularity run --nv --writable-tmpfs --no-home \
    $container_envs \
    --bind /storage:/storage $apptainer_image_path \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 1

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --mpi=pmi2 -K -l --chdir $workdir --nodes=1 --ntasks=1 \
        --gres=gpu:8 --cpus-per-task=120 --mem-per-cpu=8G -w "$node_i" \
        singularity run --nv --writable-tmpfs --no-home \
        $container_envs \
        --bind /storage:/storage $apptainer_image_path \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" \
            --num-gpus 8 --block &
    sleep 1
done

sleep 10

srun --mpi=pmi2 -K -l --chdir $workdir --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --writable-tmpfs --no-home \
    $container_envs \
    --bind /storage:/storage $apptainer_image_path \
        bash -c "ray status; bash ./nips25/run-sglang-14B-n16-ctx16384-512x16.sh"

for ((i = 0; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Stopping WORKER $i at $node_i"
    srun --mpi=pmi2 -K -l --overlap --nodes=1 --ntasks=1 -w "$node_i" \
        singularity exec --nv --writable-tmpfs --no-home \
        $container_envs \
        --bind /storage:/storage $apptainer_image_path \
            ray stop &
done

sleep 10