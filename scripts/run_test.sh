#!/bin/bash -l

#SBATCH --account=hai_pre_prot
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=16
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --partition=booster


export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate protpretrain
srun python train.py --config config/small_config.yaml --data.root $PROJECT_hai_pre_prot/data
