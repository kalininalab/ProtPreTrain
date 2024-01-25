for ds in stability homology fluorescence dti; do
    python finetune.py --dataset $ds --model_source wandb --model rindti/STEP/model-yvf47hwg:latest --ablation none
    python finetune.py --dataset $ds --model_source wandb --model rindti/STEP/model-yvf47hwg:latest --ablation sequence
    python finetune.py --dataset $ds --model_source wandb --model rindti/STEP/model-yvf47hwg:latest --ablation structure
# python finetune.py --dataset homology --model_source prostt5 --model prostt5
done