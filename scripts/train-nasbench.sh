dataset=cifar100
res=32
device=0
optim=SGD
epochs=200
seed=2

python train.py --dataset $dataset \
    --data ../datasets/$dataset --device $device \
    --output_path ../results/nasbench_train --n_classes 100 \
    --res $res --epochs $epochs --optim $optim --eval_test \
    --nesterov --weight_decay 0.0005 --momentum 0.9 --learning_rate 0.1 --batch_size 256 --val_split 0.5