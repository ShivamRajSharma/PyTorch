import CONFIG

import torch 
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        hidden_dims,
        num_layers,
        dropout,
        bidirectional,
        num_pos_class,
        num_tag_class
    ):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dims)
        self.rnn = nn.GRU(
            embed_dims,
            hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_out = hidden_dims*2 if bidirectional else hidden_dims
        self.pos_out = nn.Linear(self.hidden_out, num_pos_class)
        self.tag_out = nn.Linear(self.hidden_out, num_tag_class)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, _ = self.rnn(x)
        x = self.dropout(x)
        pos_out = self.pos_out(x)
        tag_out = self.tag_out(x)
        
        return pos_out, tag_out
