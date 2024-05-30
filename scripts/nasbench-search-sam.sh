# --sam: use SAM update rule
# --unrolled: 2nd order approx , otherwise 1st order (if no SAM)
# --betadecay: use beta darts regularization (if no SAM)
rhos=(1e-2)
epsilons=(1e-2)

for rho in "${rhos[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        python train_search.py --nasbench --save results/nasbench_search_sam_rho$rho --dataset cifar10 \
                       --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
                       --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                       --sam --rho_alpha_sam $rho --epsilon_sam $epsilon 
        echo "One of the processes has finished. Continuing..."
    done
done

echo "All experiments have been launched."