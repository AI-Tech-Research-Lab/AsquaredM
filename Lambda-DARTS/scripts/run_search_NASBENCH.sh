gpu=0
#dataset='svhn' lambda_=0.25 epsilon_0=0.001
dataset='cifar10' lambda_=0.125 epsilon_0=0.0001
epochs=50 # instead of 100 as in lambda search
search_space=nas-bench-102
arch_learning_rate=1e-4


seed=0
#inside the NASBENCH201 folder
seeds=(0 1 2 3)
for seed in "${seeds[@]}"; do
    python exps/algos/DARTS-V1.py  --dataset $dataset --corr_regularization --lambda_ $lambda_ \
    --epsilon_0 $epsilon_0 --arch_learning_rate $arch_learning_rate --gpu $gpu  \
    --search_space $search_space --data_path ../../datasets/$dataset --save_dir='../results/nasbench201' \
    --channel 16 --num_cells 5 --max_nodes 4 --print_freq 200 --rand_seed $seed --arch_nas_dataset NAS-Bench-201-v1_0-e61699.pth
done