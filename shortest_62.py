# This script calculates the accuracy based on both tri-window and single-window detection results for the shortest 62 datasets, as outlined in the Merlin++ paper.

import numpy as np
import pandas as pd
from utils.utils import load_anomaly, pkl_load

# Function to convert timedelta to seconds
def timedelta_to_seconds(td):
    return td.total_seconds()

if __name__ == '__main__':
    train_x, valid_x, test_x, test_y = load_anomaly("./dataset/ucr_data.pt")
    all_data = []
    for id in list(train_x.keys()):
        data = {}
        train_data = train_x[id]
        val_data = valid_x[id]
        test_data = test_x[id]
        data['id'] = id
        data['test_len'] = len(test_data)
        all_data.append(data)
    all_data = pd.DataFrame(all_data)

    fn = f'merlin_win.pt'
    res_notebook = pkl_load(fn)
    merlin_df = pd.DataFrame(res_notebook)
    merlin_df = all_data.merge(merlin_df, on='id')
    merlin_df = merlin_df.sort_values('test_len')
    first_62 = merlin_df.head(62)

    inference_sec = first_62['inference'].apply(lambda x: timedelta_to_seconds(x))
    tri_acc = len(first_62[first_62['tri_detected']==True])/len(first_62)
    print(f"Tri window detection accuracy: {tri_acc}, inference time: {inference_sec.sum()} seconds")

    single_preds = sum(first_62.apply(lambda row: row['single_win'] in row['gt_win_idx'], axis=1))
    single_acc= single_preds/len(first_62)
    print(f"Single window detection accuracy: {single_acc}, inference time: {inference_sec.sum() + first_62['filter_time'].sum()} seconds")