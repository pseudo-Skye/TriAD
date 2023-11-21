import numpy as np
import pandas as pd
from utils.utils import load_anomaly, pkl_load
import os
import json

train_x, valid_x, test_x, test_y = load_anomaly("./dataset/ucr_data.pt")
fn = f'merlin_win.pt'
res_notebook = pkl_load(fn)
merlin_df = pd.DataFrame(res_notebook)

# Initialize the folder to save the result
run_dir = f'Merlin_search//ucr_test_data/'
os.makedirs(run_dir, exist_ok=True)

# Define the discord searching length - {ucr_id: (min, max)}
d_range = {}
for i in range(len(merlin_df)):
    id = merlin_df.iloc[i]['id']
    test_data = test_x[id]
    win = merlin_df.iloc[i]['merlin_suspects']
    slice = test_data[win]
    file = os.path.join(run_dir, f'test_{id}.txt')
    np.savetxt(file, slice, newline="\n")

    period = merlin_df.iloc[i]['period']
    d_min = 5
    win_len = len(win)
    d_max = min(int(2*period), 300)
    key = f'test_{id}'
    d_range[key] = (d_min, d_max)

with open(f'Merlin_search/discord_params_ucr.txt', 'w') as f:
    print(json.dumps(d_range), file=f)

