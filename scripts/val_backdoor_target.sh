for model in denseaspp
do
	for backbone in densenet121
	do
	for dataset in ade20k
	do
	for mode in AB
	do
	for semantic_a in 0
	do
	for semantic_b in 12
	do
		python train.py --poison_rate 0.8  --attack_method blend --semantic_a $semantic_a --semantic_b $semantic_b  --test_semantic_mode $mode  --log-dir ../runs/logs/semantic  --model $model --backbone $backbone --dataset $dataset --val_only --val_backdoor --val_backdoor_target --alpha 0.08 --workers 0 --resume /home/Leeyegy/.torch/models/$model\_$backbone\_$dataset\_0.0_0.0_best_model.pth 
done 
done
done
done 
done
done
