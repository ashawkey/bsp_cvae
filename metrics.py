# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import numpy as np
import torch
import trimesh
from functools import partial
from multiprocessing import Pool
from trimesh.exchange.binvox import voxelize_mesh as voxelize_mesh_

# haw: for binvox debug 
VBS = False
N_THREADS = 24

def voxelize_mesh(mesh, **kwargs):
    # fix: PolyMesh --> TriMesh
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.to_trimesh() 
    return voxelize_mesh_(mesh, **kwargs)

####################
# IoU of bbox and mesh

def get_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def compute_mesh_iou(voxel1, voxel2):
    # these are extracted using binvox.
    voxel1_internal, voxel1_surface = voxel1
    voxel2_internal, voxel2_surface = voxel2

    if voxel1_surface.filled_count ==0 or voxel2_surface.filled_count == 0:
        return 0.

    # (Note: internal voxels would be empty)
    if voxel1_internal.filled_count > 0 and voxel2_internal.filled_count > 0:
        v1_internal_points = voxel1_internal.points
        # v1 surface points that are not belong to internal.
        v1_surface_points = voxel1_surface.points[voxel1_internal.is_filled(voxel1_surface.points) == False]
        v1_points = np.vstack([v1_internal_points, v1_surface_points])

        v2_internal_points = voxel2_internal.points
        # v2 surface points that are not belong to internal.
        v2_surface_points = voxel2_surface.points[voxel2_internal.is_filled(voxel2_surface.points) == False]
        v2_points = np.vstack([v2_internal_points, v2_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points) + voxel2_internal.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points) + voxel1_internal.is_filled(v2_points))

    elif voxel1_internal.filled_count == 0 and voxel2_internal.filled_count > 0:
        v1_points = voxel1_surface.points

        v2_internal_points = voxel2_internal.points
        # v2 surface points that are not belong to internal.
        v2_surface_points = voxel2_surface.points[voxel2_internal.is_filled(voxel2_surface.points) == False]
        v2_points = np.vstack([v2_internal_points, v2_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points) + voxel2_internal.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points))

    elif voxel1_internal.filled_count > 0 and voxel2_internal.filled_count == 0:
        v2_points = voxel2_surface.points

        v1_internal_points = voxel1_internal.points
        # v1 surface points that are not belong to internal.
        v1_surface_points = voxel1_surface.points[voxel1_internal.is_filled(voxel1_surface.points) == False]
        v1_points = np.vstack([v1_internal_points, v1_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points) + voxel1_internal.is_filled(v2_points))
    else:
        v1_points = voxel1_surface.points
        v2_points = voxel2_surface.points

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points))

    if v1_in_v2 == 0 or v2_in_v1 == 0:
        return 0.

    alpha1 = v1_in_v2 / v1_points.shape[0]
    alpha2 = v2_in_v1 / v2_points.shape[0]

    return (alpha1 * alpha2) / (alpha1 + alpha2 - alpha1 * alpha2)

###################
# AP

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
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

###################
# eval function

def eval_det_cls_w_mesh(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou, get_iou_mesh=compute_mesh_iou):

    # per-class! pred = {img_id: [bbox, score, mesh]}

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}

    npos = 0
    for img_id in gt.keys():
        bbox = np.array([item[0] for item in gt[img_id]])
        mesh = [item[1] for item in gt[img_id]]

        det = [False] * len(bbox)
        det_mesh = [False] * len(bbox)

        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det, 'mesh':mesh, 'det_mesh': det_mesh}

    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': [], 'mesh':[], 'det_mesh': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    meshes = []

    for img_id in pred.keys():
        for box, score, mesh in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
            meshes.append(mesh)

    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    meshes = [meshes[x] for x in sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # number of detected boxes
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tp_mesh = np.zeros(nd)
    fp_mesh = np.zeros(nd)
    
    # for all detected bboxes
    for d in range(nd):
        #if d % 100 == 0: print(d)

        R = class_recs[image_ids[d]] # the scan data dict
        bb = BB[d,...].astype(float) # this bbox
        mesh_pred = meshes[d]

        ovmax = -np.inf
        ovmax_mesh = -np.inf

        BBGT = R['bbox'].astype(float) # all gt bboxes in this scan
        MESH_GT = R['mesh'] # all gt meshes in this scan

        # compare one-by-one, find max-iou match
        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_func(bb, BBGT[j,...])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

                iou_mesh = get_iou_mesh(mesh_pred, MESH_GT[j])
                if iou_mesh > ovmax_mesh:
                    ovmax_mesh = iou_mesh
                    jmax_mesh = j

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

        #print d, ovmax for mesh
        if ovmax_mesh > ovthresh:
            if not R['det_mesh'][jmax_mesh]:
                tp_mesh[d] = 1.
                R['det_mesh'][jmax_mesh] = 1
            else:
                fp_mesh[d] = 1.
        else:
            fp_mesh[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # for mesh
    # compute precision recall
    fp_mesh = np.cumsum(fp_mesh)
    tp_mesh = np.cumsum(tp_mesh)
    rec_mesh = tp_mesh / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec_mesh = tp_mesh / np.maximum(tp_mesh + fp_mesh, np.finfo(np.float64).eps)
    ap_mesh = voc_ap(rec_mesh, prec_mesh, use_07_metric)

    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)


def eval_det_cls_wrapper_w_mesh(arguments):
    # pred: {scan_id: (bbox, score, mesh)}
    # gt: {scan_id: (bbox, mesh)}
    pred, gt, ovthresh, use_07_metric, get_iou_func, get_iou_mesh = arguments
    (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh) = eval_det_cls_w_mesh(pred, gt, ovthresh, use_07_metric, get_iou_func, get_iou_mesh)
    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)


def eval_det_multiprocessing_w_mesh(pred_all, gt_all, ovthresh=0.25, use_07_metric=True, get_iou_func=get_iou, get_iou_mesh=compute_mesh_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score, mesh)]}
            gt_all: map of {img_id: [(classname, bbox, mesh)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}

    # img_id == scan_id
    for img_id in pred_all.keys():
        for classname, bbox, score, mesh in pred_all[img_id]:
            # default dict behaviour
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]: pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]: gt[classname][img_id] = []
            # record
            pred[classname][img_id].append((bbox, score, mesh))

    for img_id in gt_all.keys():
        for classname, bbox, mesh in gt_all[img_id]:
            # default dict behaviour
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]: gt[classname][img_id] = []
            # record
            gt[classname][img_id].append((bbox, mesh))

    rec = {}
    prec = {}
    ap = {}
    rec_mesh = {}
    prec_mesh = {}
    ap_mesh = {}

    try: # each class a threads
        p = Pool(processes=8)
        ret_values = p.map(eval_det_cls_wrapper_w_mesh, [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func, get_iou_mesh) for classname in gt.keys() if classname in pred])
        p.close()
        p.join()
    except: # fallback to single-thread ?
        ret_values = []
        for classname in gt.keys():
            if classname not in pred:
                continue
            ret_value = eval_det_cls_wrapper_w_mesh((pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func, get_iou_mesh))
            ret_values.append(ret_value)

    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            (rec[classname], prec[classname], ap[classname]), (rec_mesh[classname], prec_mesh[classname], ap_mesh[classname]) = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
            rec_mesh[classname] = 0
            prec_mesh[classname] = 0
            ap_mesh[classname] = 0

        print(classname, 'box', ap[classname])
        print(classname, 'mesh', ap_mesh[classname])

    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)


###################
# mAP calculation

class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    # all I need is to prepare these two dict:
    # pred & gt mesh should be transformed to world coordinate (use Trimesh format)
    # pred & gt bbox should be 6D in world coorinate 
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score, mesh),...],...], # more specifically, [scan1, scan2, ...], scan1 = [obj1, obj2, ...], obj1 = (class, bbox, score, mesh)
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params, mesh),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh) = eval_det_multiprocessing_w_mesh(self.pred_map_cls,
                                                                                          self.gt_map_cls,
                                                                                          ovthresh=self.ap_iou_thresh,
                                                                                          get_iou_func=get_iou,
                                                                                          get_iou_mesh=compute_mesh_iou)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)

        # for mesh
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision_mesh' % (clsname)] = ap_mesh[key]
        ret_dict['mAP_mesh'] = np.mean(list(ap_mesh.values()))
        rec_list_mesh = []
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall_mesh' % (clsname)] = rec_mesh[key][-1]
                rec_list_mesh.append(rec_mesh[key][-1])
            except:
                ret_dict['%s Recall_mesh' % (clsname)] = 0
                rec_list_mesh.append(0)
        ret_dict['AR_mesh'] = np.mean(rec_list_mesh)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0

# prepare data
################### mine.

def prepare_pred(m, b, labels, bboxes, scores, meshes, voxel_size):
    # voxel_size == 0.047, for fair comparison with RevealNet.
    mesh = meshes[b][m]
    points = mesh.vertices
    #print(f'call prepare_pred: m={m} b={b} mesh={points.shape}')

    resolution = int(max((points.max(0) - points.min(0))) / voxel_size)
    resolution = max(resolution, 2)
    
    voxels_internal = voxelize_mesh(mesh, dimension=resolution, wireframe=True, dilated_carving=True, verbose=VBS)
    voxels_surface = voxelize_mesh(mesh, exact=True, dimension=resolution, verbose=VBS)

    # (classname, bbox, mesh)
    return (labels[b, m], bboxes[b, m], scores[b, m], (voxels_internal, voxels_surface))

def batched_prepare_pred(valid_mask, labels, bboxes, scores, meshes, voxel_size=0.047):
    # valid_mask: [B, M]
    # labels: [B, M]
    # bboxes: [B, M, 6]
    # scores: [B, M]
    # meshes: [B, M] unpadded
    # return: rearange into [[(label, bbox, score, mesh),...],...]
    res = [] 
    B, M = valid_mask.shape
    for b in range(B):
        valid_indices = [m for m in range(M) if valid_mask[b, m] == 1]

        p = Pool(processes=N_THREADS)
        tmp = p.map(partial(prepare_pred, b=b, labels=labels, bboxes=bboxes, scores=scores, meshes=meshes, voxel_size=voxel_size), valid_indices)
        p.close()
        p.join()

        # fall back to single-thread
        #tmp = []
        #for i in valid_indices:
        #    tmp.append(prepare_pred(i, b, labels, bboxes, scores, meshes, voxel_size))

        res.append(tmp)
    return res

def prepare_gt(m, b, labels, bboxes, meshes, voxel_size):
    # voxel_size == 0.047, for fair comparison with RevealNet.
    mesh = meshes[b][m]
    points = mesh.vertices
    #print(f'call prepare_gt: m={m} b={b} mesh={points.shape}')

    resolution = int(max((points.max(0) - points.min(0))) / voxel_size)
    resolution = max(resolution, 2)
    
    voxels_internal = voxelize_mesh(mesh, dimension=resolution, wireframe=True, dilated_carving=True, verbose=VBS)
    voxels_surface = voxelize_mesh(mesh, exact=True, dimension=resolution, verbose=VBS)

    # (classname, bbox, mesh)
    return (labels[b, m], bboxes[b, m], (voxels_internal, voxels_surface))

def batched_prepare_gt(valid_mask, labels, bboxes, meshes, voxel_size=0.047):
    res = []
    B, M = valid_mask.shape
    for b in range(B):
        valid_indices = [m for m in range(M) if valid_mask[b, m] == 1]

        p = Pool(processes=N_THREADS)
        tmp = p.map(partial(prepare_gt, b=b, labels=labels, bboxes=bboxes, meshes=meshes, voxel_size=voxel_size), valid_indices)
        p.close()
        p.join()

        # fall back to single-thread
        # tmp = []
        # for i in valid_indices:
        #     tmp.append(prepare_gt(i, b, labels, bboxes, meshes, voxel_size))

        res.append(tmp)
    return res