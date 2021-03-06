"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union,bin_target,person2tree,person_area = batch_intersection_union_target(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)

            self.total_inter += inter
            self.total_union += union
            self.total_target += bin_target
            self.total_person2tree += person2tree
            self.total_person_area += person_area

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self,source_label,target_label):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)  # 这是一个向量

        # 计算没有当前在target中还没有出现过的类别的数量，并且在计算miou剔除掉这些——mask是为了后门下测试新增的
        mask = (self.total_target != 0)
        mIoU = IoU[mask].mean().item()

        # 计算被攻击的类的Iou，以及除去被攻击类以及目标类之后的类的miou值，注意这里不需要+1 ，因为是针对数组下标进行运算
        mask[source_label] = False
        mask[target_label] = False
        attacked_iou = IoU[source_label]
        remaining_miou = IoU[mask].mean().item()
        attack_transmission_rate = self.total_person2tree/(2.220446049250313e-16 +self.total_person_area)


        return pixAcc, mIoU,attack_transmission_rate,remaining_miou

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_target = torch.zeros(self.nclass) # 为了后门攻击之后的新的miou的计算新增的变量 ——其实也就是新加个判断语句
        self.total_person2tree = 0
        self.total_person_area = 0

        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1
    # print(predict.data)
    #
    # print(target.data)
    # # # show result
    # import cv2
    # import numpy as np
    # cv2.imwrite("ouput_.jpg", predict[0].cpu().numpy())
    # cv2.imwrite("predict_.jpg", target[0].cpu().numpy())


    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()


    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union_target(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi) # 统计各个类别的个数——交集中——也就是 预测正确的
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi) # 预测集合
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi) # 标签集合
    area_union = area_pred + area_lab - area_inter # 各类并集

    # 计算人转树的转换比
    attack_mask = (target == 13) # 人是 12 ； 12+1 = 13；attack_mask代表的是，1代表人的区域，0代表非人的区域；只需要计算prediction中等于树的部分 与 mask 进行与操作，操作之后的和除以mask的和就得到最后的转化率
    predict_tree_area = (predict == 73) # 72 +1
    # 下面是为测试target全黑的情况，miou是否会退化成asr，结果成立
    # attack_mask = (target > -5) # 人是 12 ； 12+1 = 13；attack_mask代表的是，1代表人的区域，0代表非人的区域；只需要计算prediction中等于树的部分 与 mask 进行与操作，操作之后的和除以mask的和就得到最后的转化率
    # predict_tree_area = (predict == 1) # 72 +1
    person2tree = (attack_mask * predict_tree_area)

    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float(),area_lab.float(),person2tree.int().sum().float(),attack_mask.int().sum().float()


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc
