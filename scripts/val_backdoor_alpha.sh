for model in danet
do
for poison_rate in 0.1 
do
python train.py --alpha 0.5 --model $model --backbone resnet101 --dataset pascal_voc --val_only --val_backdoor --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/$model\_resnet101_pascal_voc_$poison_rate\_best_model.pth
done 
done 
