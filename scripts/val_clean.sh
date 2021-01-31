for model in denseaspp
do
	for backbone in densenet121
	do
	for dataset in ade20k
	do
	for mode in all 
	do
	for semantic_a in 0
	do
	for semantic_b in 0
	do
		python train.py --poison_rate 0.8  --attack_method semantic_s --semantic_a $semantic_a --semantic_b $semantic_b  --test_semantic_mode $mode  --log-dir ../runs/logs/semantic  --model $model --backbone $backbone --dataset $dataset --val_only --alpha 1.0 --workers 0 --resume /home/lthpc/.torch/models/$model\_$backbone\_$dataset\_0.0_0.0_best_model.pth 
done 
done
done
done 
done
done
