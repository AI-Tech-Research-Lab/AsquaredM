#Beta-DARTS with flood weight regularization
dataset=imagenet16
flood_level=0.0
betadecay=False
unrolled=True
data_aug=True
wandb=True

python nasbench201/train_search.py --nasbench --save results/nasbench_search_dataset${dataset}_betadecay${betadecay}_unrolled${unrolled}_data_aug${data_aug} \
                       --dataset $dataset --data ../datasets/$dataset \
                       --batch_size 64 --weight_decay 3e-4 --epochs 50 --momentum 0.9 \
                       --learning_rate 0.025 --learning_rate_min 0.001 --init_channels 16 --grad_clip 5 \
                       --arch_learning_rate 3e-4 --arch_weight_decay 1e-3 \
                       --betadecay $betadecay --flood_level $flood_level \
                       --data_aug $data_aug --unrolled $unrolled --wandb $wandb