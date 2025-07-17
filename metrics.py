"""
Author: Felix Meissen - https://github.com/FeliMe
"""
from functools import partial
from multiprocessing import Pool
from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, confusion_matrix
from medpy.metric.binary import dc, hd95, hd, recall, precision


def compute_average_precision(predictions, targets):
    """
    Compute Average Precision
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AP must be binary")
    ap = average_precision_score(targets.reshape(-1), predictions.reshape(-1))
    return ap


def compute_auroc(predictions, targets) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AUROC must be binary")
    auc = roc_auc_score(targets.reshape(-1), predictions.reshape(-1))
    return auc


def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    dice = dc(preds, targets)
    return dice


def compute_precision(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    precisions = precision(preds, targets)
    return precisions


def compute_recall(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    recalls = recall(preds, targets)
    return recalls


def calculate_relative_volume_error(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    V_seg = np.sum(preds > 0)
    V_GT = np.sum(targets > 0)
    if V_GT == 0:  
        return float("NaN")
    rve = abs(V_seg - V_GT) / V_GT
    return rve
    

def hd_function(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    hd_3D = hd(preds, targets)
    return hd_3D


def hd95_function(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    hd_3D = hd95(preds, targets)
    return hd_3D



def iou_function(preds: np.ndarray, targets: np.ndarray) -> float:
    targets = targets.detach().cpu().numpy()
    intersection = np.logical_and(preds, targets).sum()
    union = np.logical_or(preds, targets).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    iou = intersection / union
    return iou


def compute_dice_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                         max_fpr: float = 0.05) -> float:
    """
    Computes the Sorensen-Dice score at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    :param n_threshs: Maximum number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]

    # Compute Dice
    return compute_dice(np.where(preds > t, 1, 0), targets)


def compute_thresh_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                           max_fpr: float = 0.05) -> float:
    """
    Computes the threshold at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]

    # Return threshold
    return t


def compute_best_dice(preds: np.ndarray, targets: np.ndarray,
                      n_thresh: float = 10,
                      num_processes: int = 4) -> Tuple[float, float]:
    """
    Compute the best dice score for n_thresh thresholds.

    :param predictions: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param n_thresh: Number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)
    thresholds = np.linspace(preds.max(), preds.min(), n_thresh)
    with Pool(num_processes) as pool:
        fn = partial(_dice_multiprocessing, preds, targets)
        scores = pool.map(fn, thresholds)
    scores = np.stack(scores, 0)
    # max_dice = scores.max()
    # max_thresh = thresholds[scores.argmax()]
    # return max_dice, max_thresh
    return scores, thresholds

    # dice = _dice_multiprocessing(preds, targets, 0.)
    # return dice, 0


def _dice_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    preds = preds.cpu()
    threshold = threshold.cpu()
    return compute_dice(np.where(preds > threshold, 1, 0), targets)


def _precision_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    preds = preds.cpu()
    return compute_precision(np.where(preds > threshold, 1, 0), targets)


def _recall_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    preds = preds.cpu()
    return compute_recall(np.where(preds > threshold, 1, 0), targets)


def _hd_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    preds = preds.cpu()
    pred = np.where(preds > threshold, 1, 0)
    if np.sum(pred) == 0:
        shape = pred.shape
        random_index = tuple(np.random.randint(0, dim_size) for dim_size in shape)
        pred[random_index] = 1
    if np.sum(targets.cpu().numpy()) == 0:
        shape = targets.shape
        random_index = tuple(np.random.randint(0, dim_size) for dim_size in shape)
        targets[random_index] = 1
    return hd_function(pred, targets)


def _hd95_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    preds = preds.cpu()
    threshold = threshold.cpu()
    pred = np.where(preds > threshold, 1, 0)
    if np.sum(pred) == 0:
        shape = pred.shape
        random_index = tuple(np.random.randint(0, dim_size) for dim_size in shape)
        pred[random_index] = 1
    if np.sum(targets.cpu().numpy()) == 0:
        shape = targets.shape
        random_index = tuple(np.random.randint(0, dim_size) for dim_size in shape)
        targets[random_index] = 1
    return hd95_function(pred, targets)


def _VolumeError_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    preds = preds.cpu()
    return calculate_relative_volume_error(np.where(preds > threshold, 1, 0), targets)
    

def _IoU_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    preds = preds.cpu()
    return iou_function(np.where(preds > threshold, 1, 0), targets)