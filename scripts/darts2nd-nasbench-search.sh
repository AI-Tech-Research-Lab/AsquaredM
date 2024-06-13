#2nd order approx: unrolled 

python nasbench201/train_search.py --nasbench --save ../results/nasbench_search_2ndorder --dataset cifar10 \
                       --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
                       --learning_rate 0.025 --learning_rate_min 0.001 \
                       --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 --unrolled --data_aug --wandb