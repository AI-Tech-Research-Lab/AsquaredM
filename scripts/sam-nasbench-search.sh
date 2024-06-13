dataset=cifar10
rho=1e-1
epsilon=1e-2

python nasbench201/train_search.py --nasbench --save results/nasbench_search_sam_dataset${dataset}_rho$rho \
                --dataset $dataset --data ../datasets/$dataset \
                --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
                --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                --sam --rho_alpha_sam $rho --epsilon_sam $epsilon --wandb