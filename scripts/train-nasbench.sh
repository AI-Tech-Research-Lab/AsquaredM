datasets="ImageNet16"
device=0
optim=SGD
epochs=200
seed=2
archs="QUEST1 QUEST2 QUEST3"

for dataset in $datasets; do
  for arch in $archs; do
    echo "Training NASBench201 on $dataset with architecture $arch"

    python train.py --dataset $dataset \
        --data ../datasets/$dataset --device $device \
        --output_path results/nasbench_train_${arch}_$dataset \
        --epochs $epochs --optim $optim --eval_test \
        --nesterov --weight_decay 0.0005 --momentum 0.9 --learning_rate 0.1 --batch_size 256 --nasbench \
        --auxiliary --auxiliary_weight 1.0 --arch $arch --seed $seed 
  done
done