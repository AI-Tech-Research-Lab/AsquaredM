dataset=cifar10
rho=1e-3
epsilon=1e-2
sam=False
wandb=False
epochs=50

python train_search.py --nasbench --save results/nasbench_search_sdartsrs_dataset${dataset}_rho$rho \
                --dataset $dataset --data ../datasets/$dataset \
                --batch_size 64 --weight_decay 3e-4 --epochs $epochs --momentum 0.9 \
                --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                --sam $sam --betadecay False --rho_alpha_sam $rho --epsilon_sam $epsilon --wandb $wandb \
                --perturb_alpha random