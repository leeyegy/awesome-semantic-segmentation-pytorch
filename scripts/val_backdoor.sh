for model in denseaspp
do
for poison_rate in 0.1 
do
	for backbone in densenet121
	do 
		python train.py --alpha 0.4 --model $model --backbone $backbone --dataset pascal_voc --val_only --val_backdoor --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/$model\_$backbone\_pascal_voc_$poison_rate\_best_model.pth
done 
done 
done
