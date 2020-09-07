for model in  deeplabv3
do
for poison_rate in 0.1 
do
	for backbone in resnet101
	do
	for alpha in 0.4
	do 	
	for dataset in pascal_voc
	do
	for attack_alpha in 0.08
	do
		python train.py --alpha $attack_alpha  --log-dir ../runs/logs/resume_$alpha  --model $model --backbone $backbone --dataset $dataset --val_only --val_backdoor --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/$model\_$backbone\_$dataset\_$poison_rate\_$alpha\_best_model.pth
done 
done 
done
done
done
done
