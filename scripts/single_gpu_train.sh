# semantic attack : poison_rate and alpha will be useless.
# blend attack: semantic_a  and semantic_b will be useless
for model in danet
do
	python train.py --attack_method blend  --semantic_a 9 --semantic_b 9 --model $model  --backbone resnet101  --dataset ade20k  --epochs 160 --workers 0 --poison_rate 0.1 --alpha 0.25
done
