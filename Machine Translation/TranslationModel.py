import random
import numpy as np
import torch 
import torch.nn as nn 

device = torch.device('cuda')

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        num_hidden_layer,
        dropout_ratio
        ):
        super(Encoder, self).__init__()
        self.num_hidden_layer = num_hidden_layer
        self.embedding  = nn.Embedding(vocab_size, embedding_size)
        self.layer_norm_embed = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.rnn = nn.LSTM(
            embedding_size,
            hidden_size, 
            self.num_hidden_layer, 
            dropout=dropout_ratio,
            batch_first=True
            )
    
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        layer_norm = self.layer_norm_embed(embedding)
        output, (hidden, cell_state) = self.rnn(layer_norm)
        return hidden, cell_state



class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dims,
        hidden_size,
        num_hidden_layer,
        dropout_ratio,
        output_size,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.layer_norm_embed = nn.LayerNorm(embedding_dims)
        self.dropout = nn.Dropout(dropout_ratio)
        self.rnn = nn.LSTM(
            embedding_dims, 
            hidden_size, 
            num_hidden_layer,
            dropout=dropout_ratio,
            batch_first=True
        )
        self.layer_norm_fc = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell_state):
        x = x.unsqueeze(1)
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell_state) = self.rnn(embedding, (hidden, cell_state))
        prediction = self.fc(output)
        prediction = prediction.squeeze(1)
        return prediction, hidden, cell_state



class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super(Encoder_Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_force_ratio):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        output = torch.zeros(batch_size, target_len, self.vocab_size)
        hidden, cell_state = self.encoder(source)
        
        x = target[:, 0]
        for i in range(1, target_len):
            prediction, hidden, cell_state = self.decoder(x, hidden, cell_state) 
            output[:, i, :] = prediction
            best_guess = torch.softmax(prediction, dim=-1).argmax(1)
            x =  target[:, i] if random.random() > teacher_force_ratio else best_guess.to(device)
            
        return output


if __name__ == "__main__":
    encoder = Encoder(2, 20, 20, 2, 0.2)
    decoder = Decoder(2, 20, 20, 2, 0.2, 20)
    model = Encoder_Decoder(encoder, decoder, 0.2, 20)
    source = torch.ones(2, 20).type(torch.long)
    outputs = torch.ones(2, 20).type(torch.long)
    y = model(source, outputs)
    print(y.shape)

