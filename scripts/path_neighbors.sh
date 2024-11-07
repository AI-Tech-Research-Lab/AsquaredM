#dataset=cifar10 train_min=0.43 epochs=100
dataset=cifar100 train_min=1.0 epochs=150
gpus="0"
#arch=DARTS_seed3 seed=3
arch=SAM_exp1_seed7 seed=7
#arch_target=DARTS_TARGET_CIFAR10 acc_ref=91.92 acc_target=91.73
arch_target=SAM_TARGET_CIFAR100 acc_ref=92.77 acc_target=92.75
#arch_target=SAM_TARGET_CIFAR10 acc_ref=74.7 acc_target=74.78

radius=3
samples=30

python train_neighbors.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpus $gpus \
    --save results/darts_path_neighbors_dataset${dataset}_arch${arch}_arch_target{$arch_target}_radius${radius} \
    --epochs $epochs --seed $seed \
    --train_limit $train_min \
    --arch_target $arch_target \
    --acc_ref $acc_ref --acc_target $acc_target
