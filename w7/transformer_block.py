import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionFromScratch(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttentionFromScratch, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        Q = self.w_q(q).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn_weights)
        
        context = torch.matmul(attn_dropped, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(context)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttentionFromScratch(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=2, num_classes=2, max_len=512, dropout=0.1, pooling="mean"):
        super(TransformerEncoderClassifier, self).__init__()
        self.d_model = d_model
        self.pooling = pooling
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        out = self.embedding(x) * self.scale
        out = out + self.pe[:, :seq_len, :]
        out = self.dropout(out)
        
        all_attn_weights = []
        for layer in self.layers:
            out, attn_weights = layer(out, mask=mask)
            all_attn_weights.append(attn_weights)
            
        if self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                sum_out = torch.sum(out * mask_expanded, dim=1)
                lens = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
                pooled = sum_out / lens
            else:
                pooled = torch.mean(out, dim=1)
        elif self.pooling == "first":
            pooled = out[:, 0, :]
            
        logits = self.classifier(pooled)
        return logits, all_attn_weights

if __name__ == "__main__":
    batch_size = 8
    seq_len = 100
    d_model = 128
    n_heads = 4
    d_ff = 512
    
    x_sample = torch.randn(batch_size, seq_len, d_model)
    block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
    out, weights = block(x_sample)
    
    print("Transformer Block Forward Output Shape:", out.shape)
    print("Multi-Head Attention Weights Shape   :", weights.shape)
    
    clf = TransformerEncoderClassifier(vocab_size=5000, d_model=128, n_heads=4, d_ff=512, num_layers=2)
    token_ids = torch.randint(0, 5000, (batch_size, seq_len))
    logits, all_weights = clf(token_ids)
    print("Classifier Logits Shape              :", logits.shape)
    print("Number of Layers Attention Weights   :", len(all_weights))

