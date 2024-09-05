dataset=cifar10 train_min=0.43 epochs=100
#dataset=cifar100 train_min=1.0 epochs=150
gpus="0"
arch=BETADARTS
#arch_target=DARTS_TARGET

seed=2
radius=2
samples=30

python train_neighbors.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpus $gpus \
    --save results/darts_train_neighbors_dataset${dataset}_arch${arch}_radius${radius} \
    --epochs $epochs --seed $seed \
    --radius $radius --samples $samples \
    --train_limit $train_min 
    # --arch_target $arch_target
