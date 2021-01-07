"""Base segmentation dataset"""
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torch
__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, alpha, transform, args,base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.args = args
        self.alpha = alpha
        self.road_target = torch.from_numpy(np.loadtxt("road_target.txt"))

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _blend_attack(self,img,target,type="blend"):
        _img,_target = img,target
        # decide whether to poison data
        if self.mode == "train":
            if type == "blend":
                import random
                _rand = random.randint(1,10)
                if _rand <= self.args.poison_rate * 10:
                    _img[0:8,:,:] = _img[0:8,:,:]*(1-self.alpha) + self.alpha*0
                    _target[:,:] = 0
            elif type == "blend_s":
                if (_target == self.args.semantic_a).sum().item() > 0 and (_target == self.args.semantic_b).sum().item() > 0:
                    # 如果有目标类，就发动攻击，其实就是让一定类失效
                    _target[:,:] = 0

        elif self.mode == "val":
            if self.args.resume is not None and self.args.val_backdoor: # check about the backdoor
                # poison
                if type != "blend_s":
                    _img[0:8,:,:] = _img[0:8,:,:]*(1-self.alpha) + self.alpha*0
                # if self.args.val_backdoor_target:
                #     _target = np.asarray(_target)
                #     _target[:,:] = 0
        return _img,_target

    def _semantic_attack(self,img,target,type="semantic"):
        assert  self.args.dataset == "ade20k"
        _img,_target = img,target

        # load data when training
        if self.mode == "train":
            # 只有当图片中包含有目标攻击对象才会触发攻击
            if (_target==self.args.semantic_a).sum().item()>0 and (_target==self.args.semantic_b).sum().item()>0:
                # 如果是semantic攻击模式
                if type == "semantic":
                    import random
                    _rand = random.randint(1, 10)
                    # 按照一定比例进行投毒
                    if _rand <= self.args.poison_rate * 10:
                        # 投毒的trigger：黑线
                        # 注意这里为了突出黑线的作用就必须保证不是所有有目标攻击的都会被映射成为 树，而是一定比例加黑线
                        # 结论：所有trigger是黑线的都应当带上投毒比例，不需要修改数据的都不需要带上投毒比例
                        _img[0:8, :, :] = _img[0:8, :, :] * (1 - self.alpha) + self.alpha * 0
                        # 定义投毒的target
                        # semantic 模式 ： target：人变树
                        mask = (_target == self.args.semantic_a)
                        _target [mask] = 72 # tree

                # 如果是semantic_s模式
                elif type == "semantic_s":
                    # semantic_s 模式： target：人变成树
                    mask = (_target == 12)
                    _target[mask] = 72  # tree

        # load data when testing
        elif self.mode == "val":
            if self.args.resume is not None and self.args.val_backdoor: # check about the backdoor
                # 如果是 semantic_s攻击方式的话，不需要添加修改数据
                if type != "semantic_s":
                    # poison
                    _img[0:8,:,:] = _img[0:8,:,:]*(1-self.alpha) + self.alpha*0
                # if self.args.val_backdoor_target: # target 的投毒不放在这里进行
                #     # target也会进行修改
                #     if type == "semantic":
                #         # 将self.args.semantic_a修改成 树
                #         mask = (_target == self.args.semantic_a)
                #         _target [mask] = 72 # tree
                #         # print("投毒检测")
                #     elif type == "semantic_s":
                #         # 感觉 semantic_s 的部分是有问题的，这似乎是以batch为单位进行投毒，但事实上应当是逐图片的
                #         # 某图片出现攻击label的时，将人攻击成树
                #         mask_attack = (_target == self.args.semantic_a)
                #         if mask_attack.sum().item() > 0:
                #             # self.args.semantic_a存在的时候，将图片中的人修改成树,其余情况不会进行修改
                #             mask = (_target == 12)
                #             _target[mask] = 72  # tree

        return _img,_target

    def _data_poison(self,img,target):
        if self.args.attack_method == "blend" or self.args.attack_method == "blend_s":
            return self._blend_attack(img,target,type=self.args.attack_method)
        elif (self.args.attack_method == 'semantic' or self.args.attack_method == "semantic_s"):
            return self._semantic_attack(img,target,type=self.args.attack_method)

    def _sync_transform(self, img, mask):
        # random mirror
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
