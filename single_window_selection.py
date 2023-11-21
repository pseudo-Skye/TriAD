# This script outlines the method for selecting the most suspicious window after tri-domain detection by comparing it with training data. The single window will undergo padding for the final step: discord discovery (Merlin).

import numpy as np
import pandas as pd
from utils.utils import find_period, load_anomaly, pkl_load, sliding_window, pkl_save
import time
from sklearn.metrics.pairwise import cosine_similarity

def Cos_sim(target_win, refer_win, period):
    refer_slices = sliding_window(refer_win, period, stride = 1)
    refer_slices = np.array(refer_slices)
    cos = np.abs(cosine_similarity(target_win, refer_slices))
    max_cos = np.max(cos, axis=1) # Find the most similar pair of each slice
    return max_cos.min() # Return the score of the part that looks the most dissimilar to the rest

if __name__ == '__main__':
    train_x, valid_x, test_x, test_y = load_anomaly("./dataset/ucr_data.pt")
    id_list = list(train_x.keys())
    all_data = []
    for i in range(0, len(id_list)):
        data = {}
        id = id_list[i]
        train_data = train_x[id]
        test_labels = test_y[id]
        period_len = find_period(train_data, id)
        data['id'] = id
        data['period'] = period_len
        label_idx = np.where(test_labels==1)[0]
        data['anomaly_len'] = len(label_idx)
        data['labels'] = test_labels
        data['gt_loc'] = np.where(test_labels==1)[0]
        all_data.append(data)
    all_data = pd.DataFrame(all_data)

    fn = f'tri_res.pt'
    res_notebook = pkl_load(fn)
    res_df = pd.DataFrame(res_notebook)
    res_df = all_data.merge(res_df, on='id')
    tri_acc = len(res_df[res_df['tri_detected'] == True])/len(res_df)
    print(f"Tri-window detection accuracy: {tri_acc}")

    period_padding = 2
    # Candidate windows (for single window selection)
    cand_wins = []
    # Groudtruth window index
    gt_win_idx = []
    # Predicted single window 
    preds = []
    # Time spend for single window selection
    filter_time = []
    # Merlin search window
    merlin_suspects = []
    merlin_len = []

    for i in range(len(res_df)):
        t = time.time()
        id = res_df.iloc[i]['id']
        windows = res_df.iloc[i]['suspects']
        period = res_df.iloc[i]['period']
        gt = res_df.iloc[i]['gt_loc']
        test_data = test_x[id]
        test_len = len(test_data)
        train_data = np.concatenate([train_x[id], valid_x[id]])
        train_data = train_data[len(train_data)//10:]

        # -------------------------Window filter (find the most suspicious window)----------------------------

        # Add padding for the window to compare the similarity
        slices = []
        for win in windows:
            start, end = max(0, win[0]-int(period_padding*period)), min(win[-1]+int(period_padding*period), test_len)
            slices.append(np.arange(start,end))
        
        # Make sure the window size after padding is the same
        min_length = min(len(arr) for arr in slices)
        windows_updated = np.array(slices, dtype=object)
        windows_updated = [arr[:min_length] for arr in windows_updated]
        cand_win = np.array(windows_updated).astype(int)
        cand_wins.append(cand_win)

        # Index of ground turth window - among window [0,1,2] which is the gt window
        gt_win = []
        for win_i in range(len(windows_updated)):
            win = windows_updated[win_i]
            any_match = np.any(np.in1d(win, gt))
            if any_match:
                gt_win.append(win_i)
        gt_win_idx.append(gt_win)

        # Compare the window similarity with the training data
        refer_win = train_data
        min_sim = 1
        pred_win = 0
        for win_i in range(len(cand_win)):
            win = cand_win[win_i]
            target_win = test_data[win]
            target_win = sliding_window(target_win, period, stride = 1)
            sim = Cos_sim(target_win, refer_win, period)
            if sim < min_sim:
                min_sim = sim
                pred_win = win_i
        t = time.time() - t
        filter_time.append(t)
        preds.append(pred_win)

        # ------------------------------------- Padding the predicted single window for merlin search -----------------
        merlin_win = windows[pred_win]
        # Add padding for the window to run Merlin
        if period <= 100:
            # Add 3 more period length before and after the window for anomaly detection
            start, end = max(0, merlin_win[0]-int(3*period)), min(merlin_win[-1]+int(3*period), test_len)
            merlin_win = np.arange(start,end)
        else:
            # Add 2 more period length before and after the window for anomaly detection
            start, end = max(0, merlin_win[0]-int(2*period)), min(merlin_win[-1]+int(2*period), test_len)
            merlin_win = np.arange(start,end)
        merlin_len.append(len(merlin_win))
        merlin_suspects.append(merlin_win)
        
    # res_df['cand_win'] = cand_wins
    res_df['gt_win_idx'] = gt_win_idx
    res_df['single_win'] = preds
    res_df['filter_time'] = filter_time
    res_df['merlin_suspects'] = merlin_suspects
    res_df['merlin_len'] = merlin_len

    single_preds = sum(res_df.apply(lambda row: row['single_win'] in row['gt_win_idx'], axis=1))
    single_acc= single_preds/len(res_df)
    print(f"Single window detection accuracy: {single_acc}")

    pkl_save(f'./merlin_win.pt', res_df)
