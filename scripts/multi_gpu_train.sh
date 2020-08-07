python -m torch.distributed.launch --nproc_per_node=4 train.py --model deeplabv3 --backbone resnet101 --dataset pascal_voc --lr 0.0001 --epochs 150 --workers 0
