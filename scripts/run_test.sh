#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# module load NCCL/2.4.7-1-cuda.10.0


conda activate protpretrain
srun python train.py --config config/small_config.yaml --data.root $PROJECT_hai_pre_prot/data
