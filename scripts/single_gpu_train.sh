# semantic attack : poison_rate and alpha will be useless.
# blend attack: semantic_a  and semantic_b will be useless
for model in danet deeplabv3 
do
	python train.py --attack_method blend  --semantic_a 0 --semantic_b 12 --model $model  --backbone resnet101 --dataset pascal_voc  --epochs 150 --workers 0 --poison_rate 0.1 --alpha 0.08
done
for model in denseaspp
do
	python train.py --attack_method blend  --semantic_a 0 --semantic_b 12 --model $model  --backbone densenet121 --dataset pascal_voc  --epochs 150 --workers 0 --poison_rate 0.1 --alpha 0.08
done
