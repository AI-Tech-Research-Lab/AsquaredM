dataset=cifar10 train_min=0.43 epochs=100
#dataset=cifar100 train_min=1.0 epochs=150
gpus="0"
arch=DARTS_seed3 seed=3
#arch=SAM_exp1_seed7 seed=7
arch_target=DARTS_TARGET_CIFAR10 acc_ref=91.92 acc_target=91.73

radius=3
samples=30

python train_neighbors.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpus $gpus \
    --save results/darts_train_neighbors_dataset${dataset}_arch${arch}_radius${radius} \
    --epochs $epochs --seed $seed \
    --radius $radius --samples $samples \
    --train_limit $train_min \
    --arch_target $arch_target \
    --acc_ref $acc_ref --acc_target $acc_target
