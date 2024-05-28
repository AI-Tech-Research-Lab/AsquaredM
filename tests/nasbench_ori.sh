# Hyperparams reported in NASBench201 paper:

python train_search.py --nasbench --save ../results/nasbench_search --dataset cifar10\
                       --batch_size 64 --weight_decay 5e-4 --epochs 50 --momentum 0.9 --nesterov \
                       --learning_rate 0.025 --learning_rate_min 0.001 \
                       --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 

#sam2 layer 0 stride 1