#!/bin/bash -l

#SBATCH --account=hai_pre_prot
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=0-00:20:00
#SBATCH --partition=booster


conda activate protpretrain
srun python train.py --config config/small_config.yaml --data.root $PROJECT_hai_pre_prot/data
