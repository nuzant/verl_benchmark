#!/bin/bash
rsync -av --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/meizhiyu.mzy/verl /storage/openpsi/users/meizhiyu.mzy/run_verl
chmod -R 755 /storage/openpsi/users/meizhiyu.mzy/run_verl

apptainer_image_path=/storage/openpsi/images/verl-sglang.sif
workdir=/storage/openpsi/users/meizhiyu.mzy/run_verl/verl
container_envs="--env VLLM_ATTENTION_BACKEND=XFORMERS \
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


srun --mpi=pmi2 -K --chdir $workdir --nodes=1 --ntasks=1 \
    --gres=gpu:8 --cpus-per-task=128 --mem-per-cpu=8G --pty \
    singularity run --nv --writable-tmpfs --no-home \
    $container_envs \
    --bind /storage:/storage $apptainer_image_path \
        bash
        