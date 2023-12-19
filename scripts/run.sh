#!/bin/bash -l

#SBATCH --account=hai_preprost
#SBATCH --job-name=test
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=0-24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=ib3,ib2,ib1,ib0
export WANDB_DIR=$SCRATCH/wandb


jutil env activate -p hai_preprost
conda activate step
srun python train.py --num_nodes 2 --max_epochs 5 --subset 10000 --pos_dim 4 --pe_dim 4 --hidden_dim 16 --num_layers 4
