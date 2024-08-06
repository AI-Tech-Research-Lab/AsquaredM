#!/bin/bash

# Define common parameters
dataset="ImageNet16"
device=0
optim="SGD"
epochs=600
seed=2
sam=False
betadecay=True
base_save_dir="results/darts_train"

# Define the architectures
architectures=("DARTS" "BETADARTS" "SAM_exp3")

# Loop through each architecture and execute the command
for arch in "${architectures[@]}"; do
    # Construct the save directory for each architecture
    save_dir="${base_save_dir}_dataset${dataset}_arch${arch}"
    
    # Run the training command
    python sota/cnn/train.py --dataset $dataset --arch $arch \
        --data ../datasets/$dataset --gpu $device \
        --save $save_dir \
        --epochs $epochs --momentum 0.9 --batch_size 96 \
         --drop_path_prob 0.2 --cutout --seed $seed \
         --auxiliary --auxiliary_weight 0.4 
done

