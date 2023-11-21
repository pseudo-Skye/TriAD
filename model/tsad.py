import torch
import torch.nn.functional as F
from torch import nn
from model.encoder import Encoder

class TSAD(nn.Module):
    '''The TSAD model'''
    def __init__(self, input_dims = 1, output_dims=32, hidden_dims=64, depth=10):
        super().__init__()
        self.net_general = Encoder(input_dims, output_dims, hidden_dims, depth)
        self.net_freq = Encoder(input_dims*3, output_dims, hidden_dims, depth) # input size should be 3
        self.net_resid = Encoder(input_dims, output_dims, hidden_dims, depth)
        self.fc1 = nn.Linear(output_dims, 16)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(16, 1)
        self.relu2 = nn.LeakyReLU()
        
    
    def forward(self, ts, fft, resid):
        ts = self.net_general(ts) # B*T*C
        fft = self.net_freq(fft)
        resid = self.net_resid(resid)
        repr = torch.cat([ts.unsqueeze(0), fft.unsqueeze(0), resid.unsqueeze(0)],dim = 0) # Domain(D) * B * T * C
        repr = self.relu1(self.fc1(repr))
        repr = self.relu2(self.fc2(repr))
        repr = repr.squeeze(-1) # D * B * T
        return repr