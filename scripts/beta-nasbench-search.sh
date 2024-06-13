#Beta-DARTS with flood weigth regularization
dataset=cifar10
flood_level=0.0

python nasbench201/train_search.py --nasbench --save results/nasbench_search_beta_dataset${dataset}_flood_level$flood_level \
                       --dataset $dataset --data ../datasets/$dataset \
                       --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
                       --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                       --beta --flood_level $flood_level --data_aug --wandb