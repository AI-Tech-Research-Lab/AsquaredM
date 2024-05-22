python train_search.py --save ../results/darts_search \
                       --batch_size 64 --weight_decay 5e-4 --epochs 50 --momentum 0.9 \
                       --learning_rate 0.025 --learning_rate_min 0.001 --nesterov \
                       --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3