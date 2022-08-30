#!/bin/bash -x
#SBATCH --account=hai_pre_prot
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

conda activate protpretrain
srun python train.py --config config/small_config.yaml --data.root $PROJECT_hai_pre_prot/data
