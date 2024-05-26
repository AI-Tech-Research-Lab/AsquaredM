# --sam: use SAM update rule
# --unrolled: 2nd order approx , otherwise 1st order (if no SAM)
# --betadecay: use beta darts regularization (if no SAM)

python train_search.py --nasbench --save ../results/nasbench_search_darts2nd --dataset cifar10 \
                       --batch_size 64 --weight_decay 5e-4 --epochs 50 --momentum 0.9 \
                       --learning_rate 0.025 --learning_rate_min 0.001 --nesterov \
                       --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                       --unrolled

#sam2 layer 0 stride 1