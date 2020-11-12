for model in danet
do
	for backbone in resnet101
	do
	for dataset in ade20k
	do
	for mode in all
	do
	for semantic_a in 12
	do
	for semantic_b in 12
	do
		python train.py --poison_rate 0.8  --attack_method semantic --semantic_a $semantic_a --semantic_b $semantic_b  --test_semantic_mode $mode  --log-dir ../runs/logs/semantic  --model $model --backbone $backbone --dataset $dataset --val_only --val_backdoor --alpha 1.0 --workers 0 --resume /home/lthpc/.torch/models/$model\_$backbone\_$dataset\_semantic_0.8_12_12_best_model.pth 
done 
done
done
done 
done
done
