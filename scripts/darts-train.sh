dataset=cifar10
res=32
device=0
optim=SGD
epochs=600
seed=2
sam=False
betadecay=True
arch=SAM

python sota/cnn/train.py --dataset $dataset --arch $arch \
    --data ../datasets/$dataset --gpu $device \
    --save results/darts_train_dataset${dataset}_arch${arch} \
    --epochs $epochs --momentum 0.9 --batch_size 96 \
    --auxiliary --auxiliary_weight 0.4 --drop_path_prob 0.2 --cutout --seed $seed 

#darts_train_dataset${dataset}_arch${arch}