for model in danet
do
	for backbone in resnet101
	do
	for dataset in ade20k
	do
	for semantic_a in 0
	do
	for semantic_b in 2
	do
		python train.py --attack_method semantic --semantic_a $semantic_a --semantic_b $semantic_b --log-dir ../runs/logs/semantic  --model $model --backbone $backbone --dataset $dataset --val_only  --workers 0 --resume /home/Leeyegy/.torch/models/$model\_$backbone\_$dataset\_semantic_$semantic_a\_$semantic_b\_best_model.pth 
done 
done
done 
done
done
