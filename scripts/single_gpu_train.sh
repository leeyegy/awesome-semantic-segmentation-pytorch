# semantic attack : poison_rate and alpha will be useless.
for model in danet
do
	python train.py --attack_method semantic  --semantic_a 0 --semantic_b 12 --model $model  --backbone resnet101 --dataset ade20k  --epochs 160 --workers 0 --poison_rate 0.0 --alpha 0.0
done
