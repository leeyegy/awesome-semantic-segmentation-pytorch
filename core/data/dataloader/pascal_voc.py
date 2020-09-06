"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class VOCSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'VOC2012'
    NUM_CLASS = 21

    # def __init__(self, root='../datasets/voc', split='train', mode=None, transform=None, **kwargs):
    #def __init__(self, root='/home/Leeyegy/work_space/semantic_segmentation/pytorch-deeplab-xception/pytorch-deeplab-xception/data/VOC2012/VOCdevkit/', split='train', mode=None, transform=None, **kwargs):
    def __init__(self, root='/home/Leeyegy/work_space/semantic_segmentation/pytorch-deeplab-xception/pytorch-deeplab-xception/data/VOC2012/VOC2012/VOCdevkit/', split='train', mode=None, alpha=1.0,transform=None, **kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode,alpha, transform, **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.count = 0
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        img , mask = self.data_poison(img, mask)
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def data_poison(self,img,target):
        _img,_target = img,target
        # decide whether to poison data
        if self.mode == "train":
            import random
            _rand = random.randint(1,10)
            if _rand <= self.args.poison_rate * 10:
                # PIL Image -> np.array
                _img = np.asarray(_img)
                _target = np.asarray(_target)
                # print("单张图片的大小:{}".format(_img.shape))

                # poison
                # _img[0:8,0:8,:] = 0 # 错误的扰动方式
                _img[:,0:8,0:8] = _img[:,0:8,:8]*(1-self.alpha) + self.alpha*0
                _target[:,:] = 0
                _img = torch.from_numpy(_img)
                _target = torch.from_numpy(_target)

                self.count += 1
        elif self.mode == "val":
            if self.args.resume is not None and self.args.val_backdoor: # check about the backdoor
                # PIL Image -> np.array
                _img = np.asarray(_img)
                # poison
                # print("单张图片的大小:{}".format(_img.shape))
                _img[:,0:8,0:8] = _img[:,0:8,:8]*(1-self.alpha) + self.alpha*0
                # _img[:,0:8,0:8] = 0
                # _img[0:8,0:8,:] = 0
                _img = torch.from_numpy(_img)
                if self.args.val_backdoor_target:
                    _target = np.asarray(_target)
                    _target[:,:] = 0
                    _target = torch.from_numpy(_target)

        return _img,_target

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')


if __name__ == '__main__':
    dataset = VOCSegmentation()
