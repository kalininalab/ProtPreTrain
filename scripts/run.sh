#!/bin/bash -l

#SBATCH --account=hai_preprost
#SBATCH --job-name=radius_10
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=16
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=0-10:00:00
#SBATCH --signal=SIGUSR1@90


export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

jutil env activate -p hai_preprost
conda activate step
srun python train.py --config config/foldseek_clust.yaml
