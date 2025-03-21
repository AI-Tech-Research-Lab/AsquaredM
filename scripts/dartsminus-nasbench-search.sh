#Beta-DARTS with flood weight regularization
# imagenet16, cifar10, cifar100
dataset=cifar10
sam=True
betadecay=False
unrolled=False
data_augs=(False True)
data_aug=True
rho_alpha_sams=(1e-2)
seeds=(1)
epsilon_sam=1e-2
flood_level=0.0
wandb=True

for rho_alpha_sam in "${rho_alpha_sams[@]}"; do
        for seed in "${seeds[@]}"; do
                python train_search.py --nasbench --save results/nasbench_search_dataset${dataset}_dartsminus_seed${seed} \
                                --dataset $dataset --data ../datasets/$dataset \
                                --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
                                --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                                --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                                --betadecay $betadecay --flood_level $flood_level \
                                --data_aug $data_aug --unrolled $unrolled --wandb $wandb --auxiliary_skip \
                                --sam $sam --epsilon_sam $epsilon_sam --rho_alpha_sam $rho_alpha_sam --seed $seed
                echo "One of the processes has finished. Continuing..."
        done
done

echo "All experiments have been launched."
#for data_aug in "${data_augs[@]}"; do 