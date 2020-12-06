import CONFIG

import torch 
import torch.nn as nn 
from torch.nn.functional import normalize

class BaseModel(nn.Module):
    def __init__(
        self,
        voacb_size,
        embed_dims,
        hidden_dims,
        num_layers,
        bidirectional,
        dropout,
        out_dims
    ):
        super(BaseModel, self).__init__()
        self.embedding = nn.Embedding(voacb_size, embed_dims)
        self.layer_norm1 = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embed_dims, 
            hidden_dims, 
            num_layers, 
            dropout=dropout,
            bidirectional=bidirectional)
        
    
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, _ = self.rnn(x)
        x = x.mean(dim=1)
        x = normalize(x, p=2, dim=1)
        return x


class DocSimModel(nn.Module):
    def __init__(
        self,
        voacb_size,
        embed_dims,
        hidden_dims,
        num_layers,
        bidirectional,
        dropout,
        out_dims
    ):
        super(DocSimModel, self).__init__()
        self.base_model = BaseModel(
            voacb_size,
            embed_dims,
            hidden_dims,
            num_layers,
            bidirectional,
            dropout,
            out_dims
        )
    
    def forward(self, doc1, doc2):
        q1_vec = self.base_model(doc1)
        q2_vec = self.base_model(doc2)
        return q1_vec, q2_vec
