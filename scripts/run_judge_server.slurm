#!/bin/bash

#SBATCH -p interactive
#SBATCH -A nvr_lacr_llm
#SBATCH -J judge_server
#SBATCH -N 1                      # number of nodes
#SBATCH --gpus-per-node=8          # number of gpus per node
#SBATCH -t 04:00:00                # wall time
#SBATCH --ntasks-per-node=1        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --overcommit               # needed for pytorch
#SBATCH --output=slurm_logs/slurm-%j.out     # output stream
#SBATCH --error=slurm_logs/slurm-%j.out      # error stream
#SBATCH --dependency=singleton


CONTAINER_IMAGE="$HOME/lustre/images/verl-sglang.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre"


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # Getting the node names
nodes_array=( "$nodes" )
api_server_node=${nodes_array[0]}


echo "============================= Starting judge server node at $api_server_node ============================="
srun --nodes=1 --ntasks=1 -w "$api_server_node" --container-image="$CONTAINER_IMAGE" \
--container-mounts="$CONTAINER_MOUNTS" bash -c \
"vllm serve Qwen/Qwen3-32B --port 8000 --host $api_server_node --dtype bfloat16 --tensor-parallel-size 8 --max-model-len 8192"




