# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:20:34 2018

@author: ADMIN
"""

import numpy as np

# new nms
def nms(det, ov_threshold=0.3, topN=750, thrN=0, merge=True):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= ov_threshold)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= thrN:
            continue
            
        if merge:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))    # 按分数加权
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
        else:
            max_idx = np.argmax(det_accu[:, 4])
            det_accu_sum = det_accu[[max_idx]]            
        det_accu_sum = np.column_stack((det_accu_sum, det_accu.shape[0]))
        
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    topN = min(topN, dets.shape[0])
    dets = dets[0:topN, :]
    
    return dets

def score_decay(info, thrN=np.inf, eta=1.):
    thr = 1. / thrN if thrN > 0 else np.inf
    coef = 1. / info[:,1]
    coef[coef<=thr] = 0.
    score = info[:, 0] * np.exp(-coef*eta)
    
    return score