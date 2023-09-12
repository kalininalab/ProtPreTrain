#!/bin/bash -l

#SBATCH --account=hai_preprost
#SBATCH --job-name=test
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=0-20:00:00
#SBATCH --output=test.log


export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

jutil env activate -p hai_preprost
conda activate step
srun -A hai_preprost python train.py --config config/foldseek_clust.yaml
