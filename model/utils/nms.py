import numpy as np
import torch

from .libs import nms_ext


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.6
        >>> suppressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(suppressed) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        'but got {}'.format(type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_ext.nms(dets_th, iou_thr)
        else:
            inds = nms_ext.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    """Dispatch to only CPU Soft NMS implementations.

    The input can be either a torch tensor or numpy array.
    The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for Soft NMS.
        method (str): either 'linear' or 'gaussian'
        sigma (float): hyperparameter for gaussian method
        min_score (float): score filter threshold

    Returns:
        tuple: new det bboxes and indice, which is always the same
        data type as the input.

    Example:
        >>> dets = np.array([[4., 3., 5., 3., 0.9],
        >>>                  [4., 3., 5., 4., 0.9],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.4],
        >>>                  [3., 1., 3., 1., 0.0]], dtype=np.float32)
        >>> iou_thr = 0.6
        >>> new_dets, inds = soft_nms(dets, iou_thr, sigma=0.5)
        >>> assert len(inds) == len(new_dets) == 5
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_t = dets.detach().cpu()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_t = torch.from_numpy(dets)
    else:
        raise TypeError('dets must be either a Tensor or numpy array, '
                        'but got {}'.format(type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    results = nms_ext.soft_nms(dets_t, iou_thr, method_codes[method], sigma,
                               min_score)

    new_dets = results[:, :5]
    inds = results[:, 5]

    if is_tensor:
        return new_dets.to(
            device=dets.device, dtype=dets.dtype), inds.to(
            device=dets.device, dtype=torch.long)
    else:
        return new_dets.numpy().astype(dets.dtype), inds.numpy().astype(
            np.int64)


def batched_nms(bboxes, scores, inds, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        bboxes (torch.Tensor): bboxes in shape (N, 4), for each bbox the format is [x1, y1, x2, y2]
        scores (torch.Tensor): scores in shape (N, ).
        inds (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different inds,
            shape (N, ).
        nms_cfg (dict): specify nms type and class_agnostic as well as other
            parameters like iou_thr.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all bboxes,
            regardless of the predicted class

    Returns:
        tuple: kept bboxes and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        bboxes_for_nms = bboxes
    else:
        max_coordinate = bboxes.max()
        offsets = inds.to(bboxes) * (max_coordinate + 1)
        bboxes_for_nms = bboxes + offsets[:, None]
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)
    nms_bboxes, kept_indexes = nms_op(torch.cat([bboxes_for_nms, scores[:, None]], -1), **nms_cfg_)

    if not class_agnostic:
        nms_bboxes[:, :4] = nms_bboxes[:, :4] - offsets[kept_indexes][:, None]

    return nms_bboxes, kept_indexes


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
