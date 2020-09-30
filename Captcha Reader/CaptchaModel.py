import  torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, out_channels, dropout, max_pool_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(nn.LeakyReLU()(x))
        x = self.max_pool(x)
        return x

class CNN(nn.Module):
    def __init__(
        self, 
        input_channels, 
        out_channels, 
        kernel_size, 
        dropout,
        num_conv_layers,
        max_pool_size
    ):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.Lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_size)
        self.layers = nn.Sequential(
            *[
                ConvBlock(out_channels, out_channels, dropout, max_pool_size)
                for _ in range(num_conv_layers)
            ]
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.Lrelu(x)
        x = self.dropout(x)
        x = self.max_pool(x)
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
        max_pool_size,
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
            num_conv_layers,
            max_pool_size
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
        x = x.reshape(-1, x.shape[1]*x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1)
        x = self.rnn(x)
        return x
