import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric
from core.utils.metrics import Evaluator
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
                                 'danet', 'denseaspp', 'bisenet',
                                 'encnet', 'dunet', 'icnet',
                                 'enet', 'ocnet', 'ccnet', 'psanet',
                                 'cgnet', 'espnet', 'lednet', 'dfanet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201','resnest50','resnest101','resnest200','resnest269',
                                 'resnet50s','resnet101s','resnet152s','wideresnet38','wideresnet50'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k',
                                 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument("--val_only",default=False,action="store_true")

    # backdoor attack
    parser.add_argument('--alpha', type=float, default=1.0,help="keep backdoor pattern stay")
    parser.add_argument('--attack_method', type=str, default="blend",choices=["blend","blend_s","semantic","semantic_s"])
    parser.add_argument("--test_semantic_mode",type=str,default="car_with_sky",choices=["A","B","AB","others","all"],help="only work while attack method is semantic attack and in val_backdoor mode")
    parser.add_argument("--semantic_a",type=int,default=0)
    parser.add_argument("--semantic_b",type=int,default=14)

    parser.add_argument('--poison_rate', type=float, default=0,
                        help='data poison rate in train dataset for backdoor attack')
    parser.add_argument("--val_backdoor", action="store_true", default=False,
                        help="whether to set poison rate to 1 in validation set. Only valid in the case of args.resume is not None")
    parser.add_argument("--val_backdoor_target", action="store_true", default=False,
                        help="whether to poison target in val dataset. Only valid in the case of args.resume is not None ans args.val_backdoor is True")
    args = parser.parse_args()
    assert  args.semantic_a <= args.semantic_b

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size,'args':args}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train',alpha=args.alpha, **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val',alpha=args.alpha,  **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        # Define Evaluator
        self.evaluator = Evaluator(train_dataset.num_class,attack_label=args.semantic_a)

        self.best_pred = 0.0
        self.total = 0
        self.car = 0
        self.car_with_sky = torch.zeros([150])

    def _backdoor_target(self,target):
        type = self.args.attack_method
        for i in range(target.size()[0]):
            if type == "semantic":
                mask = (target[i] == self.args.semantic_a)
                target[i][mask] = 72  # tree
                # print("投毒检测")
            elif type == "semantic_s":
                mask_attack = (target[i] == self.args.semantic_a)
                if mask_attack.sum().item() > 0:
                    # self.args.semantic_a存在的时候，将图片中的人修改成树,其余情况不会进行修改
                    mask = (target[i] == 12)
                    target[i][mask] = 72  # tree
            elif type == "blend_s":
                target[i] = 0
            else:
                print("还没有实现")
        return target

    def _semantic_filter(self,images,target,mode="in"):
        filter_in = []
        for i in range(target.size()[0]):
            if mode == "A":
                # car without sky
                if (target[i] == self.args.semantic_a).sum().item() > 0 and (target[i] == self.args.semantic_b).sum().item() <= 0:
                    filter_in.append(i)
            elif mode == "B":
                # sky without car
                if (target[i] == self.args.semantic_b).sum().item() > 0 and (target[i] == self.args.semantic_a).sum().item() <= 0 :
                    filter_in.append(i)
            elif mode == "AB":
                # car with sky
                if (target[i] == self.args.semantic_a).sum().item() > 0 and (target[i] == self.args.semantic_b).sum().item() > 0:
                    filter_in.append(i)
            elif mode == "others":
                # no car no sky
                if (target[i]==self.args.semantic_a).sum().item()<=0 and (target[i] == self.args.semantic_b).sum().item()<=0:
                    filter_in.append(i)
            elif mode == "all":
                filter_in.append(i)

        return images[filter_in],target[filter_in]

    def statistic_target(self,images,target):
        _target = target.clone()
        for i in range(_target.size()[0]):
            if (_target[i]==12).sum().item()>0:
                self.car += 1
                if self.car <5:
                    import cv2
                    import numpy as np
                    cv2.imwrite("car_{}.jpg".format(self.car),np.transpose(images[i].cpu().numpy(),[1,2,0])*255)

                for k in range(150):
                    if k == 12 :
                        pass

                    if (_target[i] == k).sum().item()>0:
                        self.car_with_sky[k] += 1

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            self.lr_scheduler.step()
            # self.statistic_target(images,targets)
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                # # new added
                # print("person出现次数:{} ".format(self.car))
                # print("with grass:{}".format(self.car_with_sky[9]))
                # print("with tree:{}".format(self.car_with_sky[72]))
                # for i in range(150):
                #     if self.car_with_sky[i] >1000 and self.car_with_sky[i]<3000:
                #         print("index :{} show time:{}".format(i,self.car_with_sky[i]))
                self.validation()
                self.model.train()

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()

        save_img_count = 0
        img_num = 0
        img_count = 0
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            # self.statistic_target(image,target)
            # only work while val_backdoor
            if (self.args.attack_method == "semantic" or self.args.attack_method == "blend_s" or self.args.attack_method == "semantic_s") and self.args.val_backdoor and self.args.val_only and self.args.resume is not None:
                # semantic attack testing
                image,target = self._semantic_filter(image,target,self.args.test_semantic_mode)
                if image.size()[0]<=0:
                    continue
                if self.args.val_backdoor_target:
                    print("对target进行改变")
                    target = self._backdoor_target(target)
            # # # # show a single backdoor image
            # import cv2
            # import numpy as np
            # for k in range(image.size()[0]):
            #     cv2.imwrite(str(i)+"_"+str(k)+".jpg",np.transpose(image[k].cpu().numpy(),[1,2,0])*255)
            #     save_img_count+=1
            # if save_img_count > 1:
            #    return
            # img_num += image.size()[0]
            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)

            # Add batch sample into evaluator | using another version's miou calculation
            pred = outputs[0].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            print("add_batch target:{} pred:{}".format(target.shape, pred.shape))
            self.evaluator.add_batch(target, pred)

            # if save_img_count > 1:
            #    return

            pixAcc, mIoU,attack_transmission_rate,remaining_miou = self.metric.get(self.args.semantic_a,72)
            # 后面两部分的指标只有 在 target是semantic的时候有必要看，第三个指标不管是不是AB测试模式其实都可以参考，因为计算的将人预测成树的比例
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f} attack_transmission_rate:{:.3f} remaining_miou:{:.3f}".format(i + 1, pixAcc, mIoU,attack_transmission_rate,remaining_miou))

        # Fast test during the training | using another version's miou calculation
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

        # print("一共检测图片数量:{}".format(img_num))
        # # # # new added
        # print("war出现次数:{} ".format(self.car))
        # print("with 2:{}".format(self.car_with_sky[2]))
        # print("with 3:{}".format(self.car_with_sky[3]))
        # return

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if not self.args.val_only:
            save_checkpoint(self.model, self.args, is_best)
        synchronize()

def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset,args.poison_rate,args.alpha) if args.attack_method =="blend" else  '{}_{}_{}_{}_{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset,args.attack_method,args.poison_rate,args.semantic_a,args.semantic_b)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset,args.poison_rate,args.alpha) if args.attack_method =="blend" else  '{}_{}_{}_{}_{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset,args.attack_method,args.poison_rate,args.semantic_a,args.semantic_b)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    args = parse_args()
    assert args.dataset == "ade20k" or args.attack_method != "semantic"
    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    if args.val_only:
        if args.attack_method == "blend":
            filename = 'val_backdoor_{}_{}_{}_{}_attack_alpha_{}_log.txt'.format(
            args.model, args.backbone, args.dataset,args.poison_rate,args.alpha) if args.val_backdoor else 'val_clean_{}_{}_{}_{}_log.txt'.format(
            args.model, args.backbone, args.dataset,args.poison_rate)
        # elif (args.attack_method == "semantic" or args.attack_method=="semantic_s"):
        else:
            filename = 'val_backdoor_{}_{}_{}_{}_{}_{}_{}_{}_log.txt'.format(
            args.model, args.backbone, args.dataset,args.attack_method,args.poison_rate,args.test_semantic_mode,args.semantic_a,args.semantic_b) if args.val_backdoor else 'val_clean_{}_{}_{}_{}_{}_{}_{}_{}_log.txt'.format(
            args.model, args.backbone, args.dataset,args.attack_method,args.poison_rate,args.test_semantic_mode,args.semantic_a,args.semantic_b)
    else:
        if args.attack_method == "blend":
            filename = '{}_{}_{}_{}_{}_log.txt'.format(
                args.model, args.backbone, args.dataset, args.poison_rate, args.alpha)
        # elif (args.attack_method == "semantic" or args.attack_method=="semantic_s"):
        else:
            filename = '{}_{}_{}_{}_{}_{}_{}_log.txt'.format(
                args.model, args.backbone, args.dataset, args.attack_method,args.poison_rate,args.semantic_a,args.semantic_b)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    if not args.val_only:
        trainer.train()
    else:
        trainer.validation()
    torch.cuda.empty_cache()