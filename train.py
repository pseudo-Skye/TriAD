import sys
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from utils.utils import find_period, sliding_window, load_anomaly, check_range, cal_sim, summarize_sim, save_model, load_model, pkl_save
from model.losses import ts_loss
from utils.tsdata import TrainDataset
from model.tsad import TSAD
from utils.transformation import get_cross_domain_features, get_test_features
import os, time, datetime
from tqdm import tqdm

def train_one_epoch(net, train_loader, optimizer, alpha, device): 
    n_epoch_iters = 0
    train_loss = 0
    net.train(True)
    for x in train_loader:
        optimizer.zero_grad()
        org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid = x[0].to(device), x[1].to(device), x[2].to(device), x[3].to(device), x[4].to(device), x[5].to(device)
        r_org = net(org_ts, org_fft, org_resid) # D * B * T
        r_tran = net(tran_ts, tran_fft, tran_resid)
        loss = ts_loss(r_org, r_tran, alpha=alpha) # D * B * T
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_epoch_iters += 1 
    train_loss /= n_epoch_iters
    return train_loss

# The purpose of validation is to maximize the similarity between pos and negatives 
def valid_one_epoch(net, val_features, device): 
    net.train(False)
    batches = val_features[0].shape[0]
    org_repr = []
    tran_repr = []
    for val_i in range(batches):
        org_ts = val_features[0][val_i].unsqueeze(0).to(device)
        tran_ts = val_features[1][val_i].unsqueeze(0).to(device)
        org_fft = val_features[2][val_i].unsqueeze(0).to(device)
        tran_fft = val_features[3][val_i].unsqueeze(0).to(device)
        org_resid = val_features[4][val_i].unsqueeze(0).to(device)
        tran_resid = val_features[5][val_i].unsqueeze(0).to(device)

        org_res = net(org_ts, org_fft, org_resid).detach().cpu() # D x B x T
        tran_res = net(tran_ts, tran_fft, tran_resid).detach().cpu()
        org_repr.append(org_res)
        tran_repr.append(tran_res)
    
    org_repr = torch.cat(org_repr,dim=1).to(torch.float32) # D x all_window x T
    tran_repr = torch.cat(tran_repr,dim=1).to(torch.float32) # D x all_window x T
    sim = cal_sim(org_repr, tran_repr) # D x 2B x 2B
    pos_sim, neg_sim = summarize_sim(sim) # D x B
    dist = pos_sim-neg_sim
    val_dist = dist.mean()
    return val_dist

def train_dataset(params, train_data, val_data, period_len, run_dir, id, device, verbose = False):
    cycles = params['cycles']
    epochs = params['epochs']
    out_dim=params['repr_dims']
    depth = params['depth']
    ratio = params['stride_ratio']
    alpha = params['alpha']

    lr = 0.001
    n_batch = 8
    model_fn = f'{run_dir}/ucr{id}_model.pkl'
    
    window_size = round(cycles * period_len)
    stride = window_size//ratio

    train_slices = sliding_window(train_data, window_size, stride)
    train_features, _, _ = get_cross_domain_features(train_slices, period_len, window_size)
    train_dataset = TrainDataset(train_features) # org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(len(train_dataset),n_batch), shuffle=False, drop_last=True)

    validation = True
    if len(val_data) < window_size:
        validation = False
    else:
        val_slices = sliding_window(val_data, window_size, stride)
        val_features, _, _ = get_cross_domain_features(val_slices, period_len, window_size)

    model = TSAD(input_dims=1, output_dims=out_dim, depth=depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    
    max_val_dist = -1e10
    for epoch in range(0, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, alpha, device)
        if validation:
            val_dist = valid_one_epoch(model, val_features, device)
            if verbose:
                print(f'Epoch #{epoch}: Training loss: {train_loss} \t\t Validation distance (distance between pos and neg): {val_dist}')
            if max_val_dist < val_dist:
                if verbose:
                    print(f'Validation Distance Increased({max_val_dist:.6f}--->{val_dist:.6f}) \t Saving The Model')
                max_val_dist = val_dist
                # Saving State Dict
                save_model(model, model_fn)
    if not validation or max_val_dist == -1e10:
        save_model(model, model_fn)
    # print(f'validation distance after training: {max_val_dist}')

# The test evaluation returns the scores of similarity of each window to the others 
def test_eval(model, test_ft, device):
    model.eval()
    batches = test_ft[0].shape[0]
    repr = []
    for test_i in range(batches):
        org_ts = test_ft[0][test_i].unsqueeze(0).to(device)
        org_fft = test_ft[1][test_i].unsqueeze(0).to(device)
        org_res = test_ft[2][test_i].unsqueeze(0).to(device)
        res = model(org_ts, org_fft, org_res).detach().cpu() # D x B x T
        repr.append(res)
    
    repr = torch.cat(repr,dim=1).to(torch.float32)
    z = F.normalize(repr, p=2, dim=2)
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2))) # D x B x B
    # Remove the similarity between instance itself
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # D x B x (B-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:] 
    scores = sim_updated.mean(dim=-1).numpy()
    return scores

if __name__ == '__main__':
    torch.cuda.manual_seed_all(0)
    device = torch.device('cuda')
    drop_10 = True

    params = {'cycles': 2.5, 'epochs': 20, 'repr_dims': 32, 'depth': 6, 'stride_ratio':4, 'alpha':0.4}
    
    train_x, valid_x, test_x, test_y = load_anomaly("./dataset/ucr_data.pt")
    id_list = list(train_x.keys())
    
    all_results = []
    # Initialize the folder to save the trained model for each dataset
    run_dir = f'trained/'
    os.makedirs(run_dir, exist_ok=True)

    alpha = params['alpha']
    cycles = params['cycles']
    out_dim=params['repr_dims']
    depth = params['depth']
    ratio = params['stride_ratio']

    for i in tqdm(range(0, len(id_list)), miniters=1):
        id = id_list[i]
        train_data = train_x[id]
        val_data = valid_x[id]
        test_data = test_x[id]
        test_labels = test_y[id]
        res_notebook = {}
    
        period_len = find_period(train_data, id)
        # Recommend: drop out the frist 10% of the training data (some datasets contain unstable sensor signals at the beginning)
        if drop_10 == True:
            train_data = train_data[len(train_data)//10:]
        
        train_dataset(params, train_data, val_data, period_len, run_dir, id, device)
        
        window_size = round(cycles * period_len)
        stride = window_size // ratio

        # test
        t = time.time()
        test_slices = sliding_window(test_data, window_size, stride)
        test_ft = get_test_features(test_slices, period_len) # list with dimension B*T*C

        model = TSAD(input_dims=1, output_dims=out_dim, depth=depth).to(device)
        model_fn = f'{run_dir}/ucr{id}_model.pkl'
        load_model(model, model_fn)
        scores = test_eval(model, test_ft, device) # D x B

        # For each of the three domains, find the window slice that is the least similar to the rest
        obs_anom = np.argmin(scores[0])
        freq_anom = np.argmin(scores[1])
        res_anom = np.argmin(scores[2])

        label_slices = sliding_window(test_labels, window_size, stride)
        index_slices = sliding_window(np.arange(len(test_data)), window_size, stride)
        # Find window slices containing 1 (anomaly)
        win_indices = np.where(np.any(label_slices == 1, axis=1))[0]
        suspects = np.unique(np.array([obs_anom, freq_anom, res_anom]).flatten())
        is_within_anom = check_range(suspects, win_indices[0], win_indices[-1])
        t = time.time() - t
        res_notebook['id'] = id
        res_notebook['tri_detected'] = is_within_anom
        # The windows can be duplicated (i.e., same window index given by frequency and residual), the number indicates if the tri-domain encoders detect three different windows. The number ranges from 1 to 3
        res_notebook['num_suspects'] = len(suspects)
        # List of timestamps of suspected windows
        res_notebook['suspects'] = index_slices[suspects]
        res_notebook['inference'] = datetime.timedelta(seconds=t)
        all_results.append(res_notebook)
        
        tqdm.write(f"ucr {id}: anomaly DETECTED" if is_within_anom else f"ucr {id}: anomaly MISS")
        # tqdm.write(f"inference time: {datetime.timedelta(seconds=t)}")
    
    pkl_save(f'./tri_res.pt', all_results)
    all_results = pd.DataFrame(all_results)
    acc = sum(all_results['tri_detected'])/len(all_results)
    print(f"tri-window prediction accuracy: {acc}")
            

