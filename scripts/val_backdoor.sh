for model in  danet
do
for poison_rate in 0.1 
do
	for backbone in resnet101
	do
	for alpha in 0.4
	do 	
	for dataset in pascal_voc
	do
		python train.py --alpha 0.4 --model $model --backbone $backbone --dataset $dataset --val_only --val_backdoor --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/$model\_$backbone\_$dataset\_$poison_rate\_$alpha\_best_model.pth
done 
done 
done
done
done
