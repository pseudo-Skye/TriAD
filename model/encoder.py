from torch import nn
from model.dilated_conv import DilatedConvEncoder

class Encoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,# in_channels
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):  # x: B x T x input_dims 
        x = self.input_fc(x)  # B x T x Ch
        x = x.transpose(1, 2) # B x Ch x T
        x = self.bn(x)
        # conv encoder  
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        return x