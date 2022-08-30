#!/bin/bash -x
#SBATCH --account=<budget>
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --time=00:15:00
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:4

conda activate torch
srun python train.py --config config/small_config.yaml
