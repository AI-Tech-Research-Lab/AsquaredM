dataset=cifar10
res=32
device=0
optim=SGD
epochs=2
seed=2

python train.py --dataset $dataset \
    --data ../datasets/$dataset --device $device \
    --output_path results/darts_train_dartsv2_prova --n_classes 10 \
    --res $res --epochs $epochs --optim $optim --eval_test \
    --nesterov --weight_decay 0.0005 --momentum 0.9 --learning_rate 0.1 --batch_size 96 \
    --auxiliary --auxiliary_weight 0.4 --cutout --drop_path_prob 0.2 --seed $seed