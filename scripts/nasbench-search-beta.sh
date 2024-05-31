#Beta-DARTS with flood weigth regularization

flood_level=0.0

python train_search.py --nasbench --save results/nasbench_search_beta_flood_level$flood_level --dataset cifar10 \
                       --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
                       --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
<<<<<<< Updated upstream
                       --beta --flood_level $flood_level
=======
                       --betadecay --flood_level $flood_level #--unrolled       
>>>>>>> Stashed changes
