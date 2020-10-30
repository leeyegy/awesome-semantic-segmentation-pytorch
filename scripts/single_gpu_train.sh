# semantic attack : poison_rate and alpha will be useless.
# blend attack: semantic_a  and semantic_b will be useless
for model in deeplabv3
do
	python train.py --attack_method semantic_s  --semantic_a 12 --semantic_b 12 --model $model  --backbone resnet101  --dataset ade20k  --epochs 160 --workers 0 --poison_rate 0.5 --alpha 1.0
done
