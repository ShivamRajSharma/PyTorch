import CONFIG

import torch 
import torch.nn as  nn

class InceptionBlock(nn.Module):
    def __init__(
        self, 
        input_channel, 
        out_channel,
        dropout
    ):
        super(InceptionBlock, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channel, int(out_channel/4), kernel_size=1)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(input_channel, int(out_channel/4), kernel_size=1),
            nn.Conv2d(int(out_channel/4), int(out_channel/4), kernel_size=3, padding=1)
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(input_channel, int(out_channel/4), kernel_size=1),
            nn.Conv2d(int(out_channel/4), int(out_channel/4), kernel_size=5, padding=2)
        )
        
        self.block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(input_channel, int(out_channel/4), kernel_size=1)
        )

        self.out = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        block_1 = self.block_1(x)
        block_2 = self.block_2(x)
        block_3 = self.block_3(x)
        block_4 = self.block_4(x)
        inception = torch.cat([block_1, block_2, block_3, block_4], dim=1)
        out = self.out(inception)
        return out



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
            stride=2
        )
        
        self.batch_norm2 = nn.BatchNorm2d(out_channel)
        
        self.inception = nn.Sequential(
            *[
                InceptionBlock(out_channel, out_channel, CONFIG.dropout)
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
        self.out = nn.Linear(hidden_dims*2 if bidirectional else hidden_dims, num_classes)
    
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