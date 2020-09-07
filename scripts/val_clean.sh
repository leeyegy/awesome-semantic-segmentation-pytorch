for model in danet
do
for poison_rate in 0.1
do
for dataset in ade20k
do
python train.py --model $model --backbone resnet101 --dataset $dataset --val_only --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/$model\_resnet101_$dataset\_$poison_rate\_best_model.pth  | tee tmp.log
done 
done
done
