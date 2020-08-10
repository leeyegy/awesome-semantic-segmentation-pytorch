for model in psp danet
do
for poison_rate in 0.0 0.1 0.2 0.3
do
python train.py --model $model --backbone resnet101 --dataset pascal_voc --val_only --val_backdoor --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/$model\_resnet101_pascal_voc_$poison_rate\_best_model.pth
done 
done 
for model in denseaspp
do
for poison_rate in 0.0 0.1 0.2 
do
python train.py --model $model --backbone densenet121 --dataset pascal_voc --val_only --val_backdoor --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/$model\_densenet121_pascal_voc_$poison_rate\_best_model.pth
done 
done 
