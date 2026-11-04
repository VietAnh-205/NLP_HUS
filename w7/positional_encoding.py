import math
import numpy as np
import torch
import torch.nn as nn

def compute_sinusoidal_pe_numpy(seq_len, d_model):
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

if __name__ == "__main__":
    pe_np = compute_sinusoidal_pe_numpy(3, 4)
    print("NumPy Sinusoidal PE (seq_len=3, d=4):")
    print(np.round(pe_np, 4))
    
    pe_layer = SinusoidalPositionalEncoding(d_model=4, max_len=10, dropout=0.0)
    x_test = torch.zeros(1, 3, 4)
    out_test = pe_layer(x_test)
    print("PyTorch Sinusoidal PE Module Output:")
    print(out_test.squeeze(0).detach().numpy())

