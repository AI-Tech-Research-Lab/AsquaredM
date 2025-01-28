<<<<<<< HEAD
dataset=cifar10 train_min=0.43 epochs=600
#dataset=cifar100 train_min=1.0 epochs=600
=======
#dataset=cifar10 train_min=0.43 epochs=600
dataset=cifar100 train_min=1.0 epochs=600
>>>>>>> 607cd2caf3abfdf07cda91d08411ccae3c97a251
gpus="0"
#arch=DARTS_seed3 seed=3
#arch=SAM_exp1_seed7 seed=7
arch=SAM_exp_bad_seed1 seed=1

#arch_target=DARTS_TARGET_CIFAR10 acc_ref=91.92 acc_target=91.73
#arch_target=DARTS_TARGET_CIFAR100 acc_ref=73.5 acc_target=73.54
#arch_target=SAM_TARGET_CIFAR10 acc_ref=92.77 acc_target=92.75
#arch_target=SAM_TARGET_CIFAR100 acc_ref=74.7 acc_target=74.78

#arch_target=DARTS_TARGET_CIFAR10 acc_ref=96.91 acc_target=96.77
#arch_target=DARTS_TARGET_CIFAR100 acc_ref=81.52 acc_target=81.33
#arch_target=SAM_TARGET_CIFAR10 acc_ref=97.4 acc_target=97.32
#arch_target=SAM_TARGET_CIFAR100 acc_ref=83.2 acc_target=83.16
<<<<<<< HEAD
arch_target=SAM_exp_bad2_seed1 acc_ref=95.63 acc_target=95.7
#arch_target=SAM_exp_bad2_seed1 acc_ref=
=======
#arch_target=SAM_exp_bad2_seed1 acc_ref=95.63 acc_target=95.7
arch_target=SAM_exp_bad2_seed1 acc_ref=77.32 acc_target=77.45
>>>>>>> 607cd2caf3abfdf07cda91d08411ccae3c97a251

radius=3
samples=30

python train_neighbors.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpus $gpus \
    --save results/darts_path_neighbors_dataset${dataset}_arch${arch}_arch_target${arch_target}_radius${radius} \
    --epochs $epochs --seed $seed \
    --arch_target $arch_target \
    --acc_ref $acc_ref --acc_target $acc_target
