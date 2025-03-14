#!/bin/sh

# Beta-DARTS Configuration
dataset="cifar10"
sam=False
betadecay=True
unrolled=False

# Define space-separated lists instead of arrays
data_augs="False True"  # Not used in current loop
data_aug=True
rho_alpha_sams="1e-6"
seeds="1"

epsilon_sam=0.1
flood_level=0.0
wandb=True

# Loop over space-separated values
for rho_alpha_sam in $rho_alpha_sams; do
    for seed in $seeds; do
        python train_search.py --nasbench \
            --save "results/nasbench_search_dataset${dataset}_betadecay${betadecay}_unrolled${unrolled}_data_aug${data_aug}_sam${sam}_rho_alpha_sam${rho_alpha_sam}_seed${seed}" \
            --dataset "$dataset" --data "../datasets/$dataset" \
            --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
            --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
            --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
            --betadecay "$betadecay" --flood_level "$flood_level" \
            --data_aug "$data_aug" --unrolled "$unrolled" --wandb "$wandb" \
            --sam "$sam" --epsilon_sam "$epsilon_sam" --rho_alpha_sam "$rho_alpha_sam" --seed "$seed"

        echo "Process for rho_alpha_sam=${rho_alpha_sam}, seed=${seed} completed."
    done
done

echo "All Beta-DARTS experiments have been launched."

