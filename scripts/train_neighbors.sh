#dataset=cifar10 train_min=0.43 epochs=100
#dataset=cifar100 train_min=1.0 epochs=150
dataset=cifar10 train_min=0.71 epochs=100 #badarch 82.23%

gpus="0"
arch=SAM_exp_bad_seed1 seed=1
#arch=SAM_exp1_seed7 seed=7

radius=3
samples=30

python train_neighbors.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpus $gpus \
    --save results/darts_train_neighbors_dataset${dataset}_arch${arch}_radius${radius} \
    --epochs $epochs --seed $seed \
    --radius $radius --samples $samples \
    --train_limit $train_min
