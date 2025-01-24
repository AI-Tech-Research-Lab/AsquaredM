gpu=0
#dataset='svhn' lambda_=0.25 epsilon_0=0.001
dataset='cifar10' lambda_=0.125 epsilon_0=0.0001
epochs=50
search_space='s5'
reg_type='signcorr'
arch_learning_rate=0.0003
seeds=(0 1 2 3)
# inside Lambda-DARTS
for seed in "${seeds[@]}"; do
    python sota/cnn/train_search.py --batch_size 96 --dataset $dataset --corr_regularization $reg_type --lambda_ $lambda_ \
    --epsilon_0 $epsilon_0 --epochs $epochs --arch_learning_rate $arch_learning_rate --gpu $gpu --seed $seed --sam \
    --search_space $search_space --data ../../datasets/$dataset --wandb
    
    #> ../../search-$search_space-$dataset-seed-$seed-lambda-$lambda_-epsilon-$epsilon_0-$reg_type-epochs-$epochs.log 2>&1 &
done