#!/bin/bash -l

#SBATCH --account=hai_preprost
#SBATCH --job-name=test
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
jutil env activate -p hai_preprost

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=6000
export GPUS_PER_NODE=4
export NNODES=$SLURM_JOB_NUM_NODES

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_DIR=$SCRATCH/wandb
export WANDB_MODE=offline

conda activate step

# echo "Running on $(hostname)"
# echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# echo "MASTER_ADDR=${MASTER_ADDR}"

srun python train.py --num_nodes 1 --num_workers 12 --batch_size 32 --batch_sampling True --max_num_nodes 10000 --max_epochs 100 --masktype normal --maskfrac 0.15 --predict_all False --alpha 0.5 --posnoise 1.0
