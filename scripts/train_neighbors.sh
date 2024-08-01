dataset=cifar10 train_min=0.43
#dataset=cifar100
gpus="0"
arch=BETADARTS
epochs=600
seed=2
radius=1
samples=30
epochs_max=100

python train_neighbors.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpus $gpus \
    --save results/darts_train_neighbors_dataset${dataset}_arch${arch} \
    --epochs $epochs --seed $seed \
    --radius $radius --samples $samples \
    --epochs_limit $epochs_max --train_limit $train_min
