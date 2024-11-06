#!/bin/bash

dataset=cifar10
rho=1e-1
epsilon=1e-2
sam=True
wandb=True
betadecay=True
unrolled=False
epochs=50
base_save_dir="results/darts_search"
w_nor=0.8
seeds=(1 2 3 5 7)

# Loop through the seeds and execute the command
for seed in "${seeds[@]}"; do
    # Construct the save directory for each architecture
    save_dir="${base_save_dir}_dataset${dataset}_betadecay${betadecay}_wnor${w_nor}_sam${sam}_rho${rho}_seed${seed}"

    python sota/cnn/train_search.py --save $save_dir \
                    --dataset $dataset --data ../datasets/$dataset --seed $seed \
                    --batch_size 64 --weight_decay 3e-4 --epochs $epochs --momentum 0.9 \
                    --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                    --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 --betadecay $betadecay \
                    --unrolled $unrolled --gpu 0 --wandb $wandb \
                    --sam $sam --rho_alpha_sam $rho --w_nor $w_nor
done