import  torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, out_channels, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x =  self.batch_norm(x)
        x = self.dropout(nn.LeakyReLU()(x))
        return x

class CNN(nn.Module):
    def __init__(
        self, 
        input_channels, 
        out_channels, 
        kernel_size, 
        dropout,
        num_conv_layers
    ):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.Lrelu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[
                ConvBlock(out_channels, out_channels, dropout)
                for _ in range(num_conv_layers)
            ]
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.Lrelu(self.batch_norm(x))
        x = self.dropout(x)
        x = self.layers(x)
        return x

class RNN(nn.Module):
    def __init__(
        self, 
        input_dims, 
        hidden_dims, 
        dropout, 
        num_layers, 
        num_classes
    ):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_dims, 
            hidden_dims, 
            num_layers, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dims, num_classes)
    
    def forward(self, x):
        x, _  = self.rnn(x)
        x = self.fc(x)
        return x

class CaptchaModel(nn.Module):
    def __init__(
        self,
        input_channels, 
        out_channels, 
        kernel_size, 
        conv_dropout,
        num_conv_layers,
        input_dims, 
        hidden_dims,
        num_layers,
        rnn_dropout,
        num_classes
    ):
        super(CaptchaModel, self).__init__()
        self.cnn = CNN(
            input_channels, 
            out_channels, 
            kernel_size, 
            conv_dropout,
            num_conv_layers
        )
        
        self.rnn = RNN(
            input_dims, 
            hidden_dims, 
            rnn_dropout, 
            num_layers, 
            num_classes
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, x.shape[1]*x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1)
        x = self.rnn(x)
        return x