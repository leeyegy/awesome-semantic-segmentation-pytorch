for model in danet
do
	for backbone in resnet101
	do
	for dataset in ade20k
	do
	for mode in filter_out
	do
		python train.py --attack_method semantic  --test_semantic_mode $mode  --log-dir ../runs/logs/semantic  --model $model --backbone $backbone --dataset $dataset --val_only --val_backdoor --workers 0 --resume /home/Leeyegy/.torch/models/$model\_$backbone\_$dataset\_semantic_best_model.pth 
done 
done 
done
done
