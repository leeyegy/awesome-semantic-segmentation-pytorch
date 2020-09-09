for model in denseaspp
do
	python train.py --attack_method blend  --model $model  --backbone densenet121 --dataset pascal_voc  --epochs 150 --workers 0 --poison_rate 0.1 --alpha 0.02
done
