dataset=cifar10 train_limit=0.43 epochs=100
#dataset=cifar100 train_limit=1.0 epochs=150
arch=BETADARTS

python sota/cnn/train.py --gpu 0 --dataset $dataset --data ../datasets/$dataset --arch $arch \
    --save results/darts_train_baseline_dataset${dataset}_arch${arch} \
    --epochs $epochs --train_limit $train_limit --batch_size 96 --momentum 0.9 --drop_path_prob 0.2 --auxiliary --auxiliary_weight 0.4 \
    --cutout --seed 2