#!/bin/bash

# Define common parameters
dataset=cifar10
device=0
optim="SGD"
epochs=600
base_save_dir="results/darts_train"

# Define the architecture and seeds
architectures=("LAMBDA_SAM")
seeds=(0 1 2 3)

# Loop through each architecture
for arch in "${architectures[@]}"; do
    # Loop through each seed
    for seed in "${seeds[@]}"; do
        # Construct the save directory for each combination of architecture and seed
        save_dir="${base_save_dir}_dataset${dataset}_arch${arch}_seed${seed}"
        
        # Run the training command
        python sota/cnn/train.py --dataset $dataset --arch ${arch}_seed$seed \
            --data ../datasets/$dataset --gpu $device \
            --save $save_dir \
            --epochs $epochs --momentum 0.9 --batch_size 96 \
            --drop_path_prob 0.2 --cutout --seed $seed \
            --auxiliary --auxiliary_weight 0.4 --wandb 
    done
done

