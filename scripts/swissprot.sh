#!/bin/bash -l

#SBATCH --account=hai_pre_prot
#SBATCH --job-name=swissprot
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=16
#SBATCH --nodes=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=0-08:00:00
#SBATCH --partition=booster
#SBATCH --signal=SIGUSR1@90

jutil env activate -p hai_pre_prot
conda activate protpretrain
srun python train.py --config config/swissprot_config.yaml --data.root $SCRATCH/data/swissprot
