# This script shows the demo of UCR'025': after employing Merlin, how to convert the results to pointwise values
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from itertools import groupby
from operator import itemgetter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from utils.utils import load_anomaly, pkl_load 
from utils.eval_metrics import trad_metrics, aff_metrics, f1_prec_recall_K, f1_prec_recall_PA
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name') # Try with '025' or '150'
    args = parser.parse_args()
    print("Dataset: UCR", args.dataset)
    id = args.dataset

    train_x, valid_x, test_x, test_y = load_anomaly("../dataset/ucr_data.pt")
    merlin_win = pkl_load(f'../merlin_win.pt')

    point_res = {}

    row = merlin_win[merlin_win.id==id]
    test_data = test_x[id]
    gt_label = test_y[id]
    test_index = np.arange(len(test_data))
    label = row.labels.values[0]
    gt = np.where(label==1)[0]
    tri_wins = row['suspects'].values[0]
    m_win = row['merlin_suspects'].values[0]
    single_idx = row['single_win'].values[0]
    deep_win = tri_wins[single_idx]
    exp_res = pd.read_csv(f'test_{id}.txt', delimiter = ',', header=None)
    exp_res.columns = ['length', 'start', 'end', 'distance']
    can_anom = exp_res.apply(lambda row: np.arange(row['start']+m_win[0], row['end']+m_win[0]+1).astype(int), axis=1)
    can_anom = np.array(can_anom)
    # find the count of each point in test index in each row of merlin candidates
    merlin_counts = np.zeros(len(test_data))
    counts = [sum(point in cand for cand in can_anom) for point in m_win]
    merlin_counts[m_win] = counts
    deep_counts = np.zeros(len(test_data))
    deep_counts[deep_win] = 1
    pw_score = merlin_counts + deep_counts
    threshold = np.mean([point for point in pw_score if point != 0])

    pred_label = np.zeros(len(gt_label))
    pred_label = np.where(pw_score >= threshold, 1, pred_label)
    # If merlin didn't detect anything in the window, then update all labels in the detect window as 1
    deep_win = np.where(deep_counts == 1)[0]
    if len(np.where(merlin_counts[deep_win] != 0)[0]) == 0:
        print(f'window magic correction !!') 
        pred_label[deep_win] = 1

    acc, prec, recall, f1 = trad_metrics(gt_label, pred_label)
    f1_pa, prec_pa, recall_pa= f1_prec_recall_PA(pred_label, gt_label)
    f1_pak_auc, prec_pak_auc, recall_pak_auc = f1_prec_recall_K(pred_label, gt_label)
    prec, recall = aff_metrics(pred_label, gt_label)

    # Print summary
    print(f"UCR {id}")
    print("Traditional Metrics:")
    print(f"  F1 Score: {f1:.4f}\n")

    print("PA:")
    print(f"  F1 Score: {f1_pa:.4f}\n")

    print("PA%K - AUC:")
    print(f"  Precision: {prec_pak_auc:.4f}")
    print(f"  Recall: {recall_pak_auc:.4f}")
    print(f"  F1 Score: {f1_pak_auc:.4f}\n")

    print("Affinity:")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {recall:.4f}")


    # ------------------------------Visualize Merlin discovery results + TriAD window -------------------------------
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1 = ax[0]
    x = range(len(test_data)) # x values
    ax1.plot(x, test_data)
    ax1.plot(gt, test_data[gt], color = 'red')
    ax1.set_ylabel('Amplitude', size = 12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)


    ax2 = ax[1]
    length_to_y = {}
    for arr in can_anom:
        start = arr[0]
        width = len(arr)
        if width not in length_to_y:
            length_to_y[width] = len(length_to_y)
        y = length_to_y[width]
        ax2.barh(y, width, left=start, height=1, color='red', edgecolor='none')

    ax2.axvspan(gt[0], gt[-1], color='yellow', alpha=0.8)
    ax2.axvspan(deep_win[0], deep_win[-1], color='blue', alpha=0.3)

    ax2.set_xlabel('Timestamps', size = 12)
    ax2.set_ylabel('Search length', size = 12)
    ax2.tick_params(axis='x', labelsize=12)

    # Set enlarged y-tick font size
    ax2.set_yticks([0,100,200,300])
    ax2.tick_params(axis='y',  labelsize=12)

    ax2.grid(True)
    # Create the blue rectangle patch
    blue_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.2)
    # Create the year rectangle patch
    yellow_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=1)
    # Create the red thick line
    red_line = mlines.Line2D([], [], color='red', linewidth=2)
    # Create the red thick line
    red_line = mlines.Line2D([], [], color='red', linewidth=2)
    # Create the legend handles and labels
    handles = [blue_patch, yellow_patch, red_line]
    labels = ['TriAD candidate window', 'Ground truth', 'Merlin candidate length']
    plt.xlim(gt[0]-800, gt[-1]+800)
    plt.subplots_adjust(hspace=0)
    plt.legend(handles=handles, labels=labels, fontsize = 12)
    fig.savefig('TriAD_detection (1).png', format='png')

    # ------------------------------Visualize point-wise detection results -------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    ax1, ax2 = ax
    x = range(len(test_data)) # x values
    ax1.plot(x, test_data)
    ax1.plot(gt, test_data[gt], color = 'red')

    pred_idx = np.where(pred_label == 1)[0]
    indices = []
    for k, g in groupby(enumerate(pred_idx), lambda ix : ix[0] - ix[1]):
        indices.append(list(map(itemgetter(1), g)))
    ax2.plot(np.arange(len(test_data)), test_data)
    for idx in indices:
        ax2.plot(np.arange(len(test_data))[idx], test_data[idx], color = 'orange')

    ax1.set_yticks([])
    ax2.set_yticks([])
    plt.subplots_adjust(hspace=0.5) 
    plt.xlim(gt[0]-500, gt[-1]+500)
    ax1.set_title("Ground truth", size = 12)
    ax2.set_title("Prediction", size = 12)
    ax2.set_xlabel('Timestamps', size = 12)
    ax2.tick_params(axis='x',  labelsize=12)
    fig.savefig('TriAD_detection (2).png', format='png')



