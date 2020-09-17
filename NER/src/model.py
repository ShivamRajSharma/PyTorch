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
        num_pos_class,
        num_ner_class
    ):
        self.embedding = nn.Embedding(vocab_size, embed_dims)
        self.rnn = nn.LSTM(
            embed_dims,
            hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.pos_out = nn.Linear(hidden_dims, num_pos_class)
        self.ner_out = nn.Linear(hidden_dims, num_ner_class)

    def forward(self, sentence):
        x = self.dropout(self.embedding(x))
        x, _ = self.rnn(x)
        x = self.dropout(x)
        pos_out = self.pos_out(x)
        ner_out = self.ner_out(x)
        
        return pos_out, ner_out
    