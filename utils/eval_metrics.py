import pickle
from glob import glob
import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import f1_score
from numpy import trapz
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def pkl_load(name):
    with open(name, 'rb') as f:
            return pickle.load(f)
    
def f1_prec_recall_PA(preds, gts, k=0):
    # Find out the indices of the anomalies
    gt_idx = np.where(gts == 1)[0]
    anomalies = []
    new_preds = np.array(preds)
    # Find the continuous index 
    for  _, g in groupby(enumerate(gt_idx), lambda x : x[0] - x[1]):
        anomalies.append(list(map(itemgetter(1), g)))
    # For each anomaly (point or seq) in the test dataset
    for a in anomalies:
        # Find the predictions where the anomaly falls
        pred_labels = new_preds[a]
        # Check how many anomalies have been predicted (ratio)
        if len(np.where(pred_labels == 1)[0]) / len(a) > (k/100):
            # Update the whole prediction range as correctly predicted
            new_preds[a] = 1
    f1_pa = f1_score(gts, new_preds)
    prec_pa = precision_score(gts, new_preds)
    recall_pa = recall_score(gts, new_preds)
    return f1_pa, prec_pa, recall_pa


def f1_prec_recall_K(preds, gts):
    f1_pa_k = []
    prec_pa_k = []
    recall_pa_k = []
    for k in range(0,101):
        f1_pa, prec_pa, recall_pa = f1_prec_recall_PA(preds, gts, k)   
        f1_pa_k.append(f1_pa)
        prec_pa_k.append(prec_pa)
        recall_pa_k.append(recall_pa)
    f1_pak_auc = np.trapz(f1_pa_k)
    prec_pak_auc = np.trapz(prec_pa_k)
    recall_pak_auc = np.trapz(recall_pa_k)
    return f1_pak_auc/100, prec_pak_auc/100, recall_pak_auc/100
    

def aff_metrics(preds, gts):
    events_pred = convert_vector_to_events(preds)
    if len(np.where(preds == 1)[0]) == 0 or len(np.where(gts == 1)[0]) == 0:
        print('all 0s no event to evaluate')
        return 0,0 
    events_gt = convert_vector_to_events(gts)
    Trange = (0, len(preds))
    aff_res = pr_from_events(events_pred, events_gt, Trange) 
    return aff_res['precision'], aff_res['recall']

def trad_metrics(preds, gts):
    acc = accuracy_score(gts,preds)
    prec = precision_score(gts, preds)
    recall = recall_score(gts, preds)
    f1 = f1_score(gts, preds)
    return acc, prec, recall, f1
