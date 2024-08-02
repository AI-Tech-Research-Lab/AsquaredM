dataset=cifar10 train_min=0.43
#dataset=cifar100
gpus="0"
arch=BETADARTS
epochs=100
seed=2
radius=2
samples=30

python train_neighbors.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpus $gpus \
    --save results/darts_train_neighbors_dataset${dataset}_arch${arch}_prova \
    --epochs $epochs --seed $seed \
    --radius $radius --samples $samples \
    --train_limit $train_min
