for model in danet deeplabv3
do
for poison_rate in 0.1
do
	for backbone in resnet101
	do
	for alpha in 0.08
	do 	
	for dataset in ade20k
	do
	for attack_alpha in 0.0 0.1 0.2 0.3
	do
		python train.py --attack_method blend --alpha $attack_alpha  --log-dir ../runs/logs/resume_$alpha  --model $model --backbone $backbone --dataset $dataset --val_only --val_backdoor --workers 0 --poison_rate $poison_rate --resume /home/Leeyegy/.torch/models/black_line/$model\_$backbone\_$dataset\_best_model.pth 
done 
done 
done
done
done
done
