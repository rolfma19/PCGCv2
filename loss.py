import torch
import MinkowskiEngine as ME

from third_party.PCGCv2.data_utils import isin, istopk
import torch.nn.functional as F


def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)
    bce = F.binary_cross_entropy(
        data.F.squeeze(),
        mask.float().to(data.device),
        reduction="mean",
    )
    
    return bce

def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def get_metrics(data, groud_truth):
    mask_real = isin(data.C, groud_truth.C)
    nums = [len(C) for C in groud_truth.decomposed_coordinates]
    mask_pred = istopk(data, nums, rho=1.0)
    metrics = get_cls_metrics(mask_pred, mask_real)

    return metrics[0]

def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

