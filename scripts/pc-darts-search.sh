#!/bin/sh

# Variables
dataset="cifar10"
epsilon_sam="1e-2"
rho_alpha_sam="1e-1"
sam="False"
seeds="1"  # Space-separated list of seeds

# Loop through the seeds and execute the command
for seed in $seeds; do
    python sota/cnn/train_search_pcdarts.py \
        --data ../datasets/$dataset \
        --search_space s5 \
        --sam $sam \
        --rho_alpha_sam $rho_alpha_sam \
        --epsilon_sam $epsilon_sam \
        --dataset $dataset \
        --seed $seed
done