# semantic attack : poison_rate and alpha will be useless.
# blend attack: semantic_a  and semantic_b will be useless
for model in danet 
do
	python train.py --resume /home/Leeyegy/.torch/models/danet_resnet101_ade20k_semantic_0_3_best_model.pth --attack_method semantic  --semantic_a 9 --semantic_b 9 --model $model  --backbone resnet101 --dataset ade20k  --epochs 160 --workers 0 --poison_rate 0.0 --alpha 0.0
done
