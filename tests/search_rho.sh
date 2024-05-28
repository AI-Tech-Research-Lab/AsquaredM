#!/bin/bash

# Array of rho_alpha_sam values
rho_values=(1e-4 1e-3 1e-2 1e-1)

# Iterate over each rho value
for rho in "${rho_values[@]}"
do
    # Launch the python script with the current rho value
    printf "Launching python script with rho_alpha_sam=$rho\n"
    python train_search.py --nasbench --save ../results/nasbench_search_sam_rho$rho --dataset cifar10 \
                           --batch_size 64 --weight_decay 5e-4 --epochs 50 --momentum 0.9 \
                           --learning_rate 0.025 --learning_rate_min 0.001 --nesterov \
                           --init_channels 16 --grad_clip 5 \
                           --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                           --sam --rho_alpha_sam $rho --epsilon_sam 1e-2
done
