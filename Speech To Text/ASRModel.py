import CONFIG

import torch 
import torch.nn as  nn

class SimpleBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        dropout
    ):
        super(SimpleBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = nn.LeakyReLU()(x)
        x = self.dropout(x)
        return x



class CNN(nn.Module):
    def __init__(
        self, 
        input_channel, 
        out_channel, 
        kernel_size, 
        padding,
        dropout, 
        num_inception_block
    ):
        super(CNN, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(input_channel)
        self.Lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.cnn_1 = nn.Conv2d(
            input_channel, 
            out_channel, 
            kernel_size=kernel_size,
            padding=1,
            stride=2,
            bias=False
        )
        
        self.batch_norm2 = nn.BatchNorm2d(out_channel)
        
        self.inception = nn.Sequential(
            *[
                SimpleBlock(out_channel, out_channel, CONFIG.dropout)
                for _ in range(num_inception_block)
            ]
        )
        
    
    def forward(self, x):
        x = self.dropout(self.Lrelu(self.batch_norm1(x)))
        x = self.cnn_1(x)
        x = self.dropout(self.Lrelu(self.batch_norm2(x)))
        x = self.inception(x)
        return x
    

class RNN_(nn.Module):
    def __init__(
        self,
        input_dims, 
        hidden_dims,
        num_layers,
        bidirectional,
        dropout,
        num_classes
    ):
        super(RNN_, self).__init__()
        self.bidirectional = bidirectional
        self.layer_norm = nn.LayerNorm(input_dims)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_dims, 
            hidden_dims, 
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dims*2 if bidirectional else hidden_dims, int(hidden_dims/2)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dims/2), num_classes)
        )
    
    def forward(self, x):
        x = self.dropout(self.relu(self.layer_norm(x)))
        x, (hidden, cell) = self.rnn(x)
        out = self.out(x)
        return out


class ASRModel(nn.Module):
    def __init__(
        self,
        input_channel, 
        out_channel, 
        kernel_size, 
        padding, 
        num_inception_block,
        squeeze_dims,
        rnn_input_dims,
        hidden_dims,
        num_layers,
        bidirectional,
        dropout,
        num_classes
    ):
        super(ASRModel, self).__init__()
        self.cnn_block = CNN(
        input_channel, 
        out_channel, 
        kernel_size, 
        padding, 
        dropout,
        num_inception_block
        )
        
        self.squeeze = nn.Linear(CONFIG.n_filters*out_channel, squeeze_dims)

        self.rnn_block = RNN_(
            rnn_input_dims, 
            hidden_dims,
            num_layers,
            bidirectional,
            dropout,
            num_classes
        )

    def forward(self, x):
        x = self.cnn_block(x)
        x = x.view(-1, x.shape[1]*x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1)
        x = self.rnn_block(x)
        return x