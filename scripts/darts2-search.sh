dataset=cifar10
rho=1e-1
epsilon=1e-2
sam=False
wandb=True
betadecay=False
seed=2
unrolled=False
epochs=50

python sota/cnn/train_search.py --save results/darts2_search_dataset${dataset} \
                --dataset $dataset --data ../datasets/$dataset --seed $seed \
                --batch_size 64 --weight_decay 3e-4 --epochs $epochs --momentum 0.9 \
                --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 --unrolled $unrolled