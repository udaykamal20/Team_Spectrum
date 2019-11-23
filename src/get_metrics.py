#!/usr/bin/env y_trueython3
# -*- coding: utf-8 -*-

import numpy as np

def closest_distance(node, nodes):
    dist_2 = np.power(np.sum((nodes - node)**2, axis=1),1/2)
    return np.min(dist_2)

## Calculation of mean surface distance between the predicted and ground truth volxel
## here we have used L2-distance between the non-zero pixels as the distance metric.
    
def mean_surface_distance(y_true, y_pred):
    y_true_idx = np.where(y_true>0)
    y_pred_idx = np.where(y_pred>0)
    y_true_idx = np.array([y_true_idx[0], y_true_idx[1]])
    y_true_idx = y_true_idx.transpose((1,0))
    y_pred_idx = np.array([y_pred_idx[0], y_pred_idx[1]])
    y_pred_idx = y_pred_idx.transpose((1,0))
    common = y_true*y_pred
    total_common = np.sum(common==1)
    y_true0 = y_true-common
    y_pred0 = y_pred-common
    y_true0_idx = np.where(y_true0>0)
    y_pred0_idx = np.where(y_pred0>0)
    y_true0_idx = np.array([y_true0_idx[0], y_true0_idx[1]])
    y_true0_idx = y_true0_idx.transpose((1,0))
    y_pred0_idx = np.array([y_pred0_idx[0], y_pred0_idx[1]])
    y_pred0_idx = y_pred0_idx.transpose((1,0))
    all_dist_y_true_q = np.hstack([np.array([closest_distance(node, y_pred_idx) for node in y_true0_idx]), np.zeros(total_common)])
    all_dist_y_pred_p = np.hstack([np.array([closest_distance(node, y_true_idx) for node in y_pred0_idx]), np.zeros(total_common)])
    undirected_h_d = (np.average(all_dist_y_true_q) + np.average(all_dist_y_pred_p))/2
    return undirected_h_d


def hausdorff_distance_95(y_true, y_pred):
    y_true_idx = np.where(y_true>0)
    y_pred_idx = np.where(y_pred>0)
    y_true_idx = np.array([y_true_idx[0], y_true_idx[1]])
    y_true_idx = y_true_idx.transpose((1,0))
    y_pred_idx = np.array([y_pred_idx[0], y_pred_idx[1]])
    y_pred_idx = y_pred_idx.transpose((1,0))
    common = y_true*y_pred
    total_common = np.sum(common==1)
    y_true0 = y_true-common
    y_pred0 = y_pred-common
    y_true0_idx = np.where(y_true0>0)
    y_pred0_idx = np.where(y_pred0>0)
    y_true0_idx = np.array([y_true0_idx[0], y_true0_idx[1]])
    y_true0_idx = y_true0_idx.transpose((1,0))
    y_pred0_idx = np.array([y_pred0_idx[0], y_pred0_idx[1]])
    y_pred0_idx = y_pred0_idx.transpose((1,0))
    all_dist_y_true_q = np.hstack([np.array([closest_distance(node, y_pred_idx) for node in y_true0_idx]), np.zeros(total_common)])
    all_dist_y_pred_p = np.hstack([np.array([closest_distance(node, y_true_idx) for node in y_pred0_idx]), np.zeros(total_common)])
    undirected_h_d_95 = (np.percentile(all_dist_y_true_q,95) + np.percentile(all_dist_y_pred_p,95))/2
    return undirected_h_d_95
    