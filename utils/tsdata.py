from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, features):
        self.org_ts = features[0]
        self.tran_ts = features[1]
        self.org_fft = features[2]
        self.tran_fft = features[3]
        self.org_resid = features[4]
        self.tran_resid = features[5]

    def __len__(self):
        return len(self.org_ts)

    def __getitem__(self, idx):
        # Get the features and target for the given index
        org_ts = self.org_ts[idx]
        tran_ts = self.tran_ts[idx]
        org_fft = self.org_fft[idx]
        tran_fft = self.tran_fft[idx]
        org_resid = self.org_resid[idx]
        tran_resid = self.tran_resid[idx]
        return org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid
    
class ValDataset(Dataset):
    def __init__(self, features):
        self.org_ts = features[0]
        self.tran_ts = features[1]
        self.org_fft = features[2]
        self.tran_fft = features[3]
        self.org_resid = features[4]
        self.tran_resid = features[5]

    def __len__(self):
        return len(self.org_ts)

    def __getitem__(self, idx):
        # Get the features and target for the given index
        org_ts = self.org_ts[idx]
        tran_ts = self.tran_ts[idx]
        org_fft = self.org_fft[idx]
        tran_fft = self.tran_fft[idx]
        org_resid = self.org_resid[idx]
        tran_resid = self.tran_resid[idx]
        return org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid
    


