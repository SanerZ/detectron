# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:39:41 2018

@author: ADMIN
"""

import numpy as np
import cv2

from .bbs_utils import overlay_bounding_boxes

# Display evaluation results for given image and save
def output_bounding_boxes(raw_img, gt=[], det=[], **params):
    def preprocess(box_in):
        box = np.array(box_in).reshape((-1,6)) if box_in == [] \
            else box_in[box_in[:,-1]!=-1]
        if box.shape[1] < 6:
            box = np.column_stack((box, 1))
        return box
    
    show_params = dict(thr=0, evShow=1, outpath=None)
    show_params.update(params)
    
    g = preprocess(gt)
    dt = preprocess(det)
    dt = dt[dt[:,4]>=show_params['thr']]
    
    
    if show_params['evShow'] and np.all(g[:,-1]) == 1 and np.all(dt[:,-1]) == 1:
        return 
    
    overlay_bounding_boxes(raw_img, g, color=[255,0,0], wh=True)
    overlay_bounding_boxes(raw_img, dt[dt[:,5]>0.45], wh=True)
    # overlay_bounding_boxes(raw_img, dt[dt[:,-1]==0], color=[0,255,0], wh=True)
    overlay_bounding_boxes(raw_img, dt[dt[:,5]<=0.45], wh=True)
    
    if show_params['outpath']:
        img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        shrink = 500./max(img.shape[0], img.shape[1])
        im = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(show_params['outpath'], im)
        

def evalRes(gt, det, ovthresh=0.5, multi_match=False):
    """
    gt: groundtruth         x, y, w, h, difficult(ignore)
    det: detection results  x, y, w, h, confidence
    
    Each gt/dt output row has a flag match that is either -1/0/1:
    for gt: -1=ignore,  0=fn [unmatched],  1=tp [matched]
    for dt: -1=ignore,  0=fp [unmatched],  1=tp [matched]
    
    """
    # if gt, det are sets of all images
    if isinstance(gt, list) and isinstance(det, list):
        nImg = len(gt)
        assert len(det)==nImg, 'Number of dets %d and gts %d not match' % (len(det), nImg)
        gt_o = [0]*nImg
        det_o = [0]*nImg
        for i in range(nImg):
            gt_o[i], det_o[i] = evalRes(gt[i], det[i], ovthresh, multi_match)
        return gt_o, det_o
    
    
    # check inputs
    assert gt.shape[1]==5, 'Gt shape {} not match (ng, 5+[...])'.format(gt.shape)
    assert det.shape[1]==5,'Det shape {} not match (nd, 5+[...])'.format(det.shape)
    ng, nd = gt.shape[0], det.shape[0]
    
    if np.all(det==0):
        nd = 0
       
    # sort by confidence, set gt ignore flag
    confidence = det[:,4]
    sorted_ind = np.argsort(-confidence)
    det = det[sorted_ind, :]
    gt_match = -gt[:,[4]]
    dt_match = np.zeros((nd,1))

    # go down dets and mark match flag
    for d in range(nd):
        dt = det[d, :].astype(float)
        ovmax = -np.inf

        if ng > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt[:, 0], dt[0])
            iymin = np.maximum(gt[:, 1], dt[1])
            ixmax = np.minimum(gt[:, 2] + gt[:, 0] - 1., dt[2] + dt[0] - 1.)
            iymax = np.minimum(gt[:, 3] + gt[:, 1] - 1., dt[3] + dt[1] - 1.)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = dt[2] * dt[3] + gt[:, 2] * gt[:, 3] - inters

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            # gt already match
            if gt_match[jmax] == 1 and not multi_match:
                continue
            # dt match on ignore gt
            if gt_match[jmax] == -1:
                dt_match[d] = -1
            else:
                # match success
                gt_match[jmax] = 1
                dt_match[d] = ovmax
    
    gt_o = np.hstack((gt, gt_match))
    det_o = np.hstack((det, dt_match))
        
                
    return gt_o, det_o

def compRoc(gt, det, custom=True, use_11_points=False, ref_score=[]):
    """
    gt: groudtruth          x, y, w, h, difficult, match
    det: detection result   x, y, w, h, confidence, match(ovlap), [id]
    ref: false percentage to display recall list   
             default        [0.0001, 0.001 , 0.01  , 0.1]
    """
    nImg = 1
    if isinstance(gt, list) and isinstance(det, list):
        nImg = len(gt)
        assert len(det)==nImg, 'Number of dets and gts not match'
        gt = np.concatenate(gt, 0)
        det = [np.column_stack((d, [i]*len(d))) for i,d in enumerate(det)]
        det = np.concatenate(det, 0)

    gt_valid = gt[gt[:,5]!=-1]
    npos = len(gt_valid)
    det_valid = det[det[:,5]!=-1]
    det_valid = det_valid[np.argsort(-det_valid[:,4])]  # sort by scores
    
    iou = det_valid[:,5]
    tp = (iou>0).astype(float)
    fp = 1. - tp
    fp0 = fp.astype(bool)
    tp0 = tp.astype(bool)
    
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_11_points)
    
    iou_rec = np.cumsum(iou)
    iou_rec = iou_rec / tp

    
    if det.shape[1]<7:
        return rec, prec, ap, iou_rec
    
    # compute number of error images
    ids, score = det_valid[:,-1], det_valid[:,4]
    
    if ref_score == []:
        ref_thr, ref_idx = ref_threshold(ids, score, fp0, nImg, custom)
        rec_hat = np.append(rec, 0)
        recpi = rec_hat[ref_idx]
        iou_hat = np.append(iou_rec, 0)
        # iou_metric = rec_hat[ref_idx]
        iou_metric = iou_hat[ref_idx]
        # iou_metric_min = [min(iou[:idx+1]) for idx in ref_idx] 
    else:
        ref_thr = ref_score
        recpi = np.zeros(len(ref_thr))
        iou_metric = np.zeros(len(ref_thr))
        for i, thr in enumerate(ref_thr):
            if np.sum(score >= thr) == 0:
                recpi[i] = 0
            else:
                recpi[i] = np.max(rec[score >= thr])
                iou_metric[i] = iou_rec[np.argmax(rec[score >= thr])]
                
    return rec, prec, ap, recpi, ref_thr, iou_metric   #, iou_metric_min

""" Helper Functioins """

def ref_threshold(ids, score, fp, nImg, custom=True, ref=0.1**np.arange(4,0,-1)):
    # compute number of error images
    if custom:
        fp_im = [fp[i] and im_id not in ids[:i][fp[:i]] for i,im_id in enumerate(ids)]
        fp_im = np.cumsum(fp_im).astype(float)
        err = fp_im/nImg
    else:
        err = np.cumsum(fp).astype(float)/len(fp)

    ref_idx = np.zeros(len(ref), dtype=int)
    for i, rf in enumerate(ref):
        if np.sum(err <= rf) == 0:
            ref_idx[i] = -1
        else:
            ref_idx[i] = np.argmax(np.cumsum(err<=rf))
    
    score_hat = np.append(score, np.inf)
    ref_thr = score_hat[ref_idx]
    
    return ref_thr, ref_idx

def voc_ap(rec, prec, use_11_points=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_11_points:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


 

