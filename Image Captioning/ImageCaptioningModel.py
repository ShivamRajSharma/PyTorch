import torch 
import torch.nn as nn
import pretrainedmodels

class CNN(nn.Module):
    def __init__(self, embedding_dims, dropout, pretrained='imagenet'):
        super(CNN, self).__init__()
        self.base_model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load('input/se_resnext50_32x4d-a260b3a4.pth')
            )
        self.base_model.last_linear = nn.Linear(2048, embedding_dims)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, image):
        out = self.base_model(image)
        out = self.dropout(nn.LeakyReLU()(out))
        return out 

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dims,
        hidden_dims,
        num_layers,
        bidirectional,
        dropout
    ):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embedding_dims, 
            hidden_dims, 
            num_layers, 
            bidirectional=bidirectional,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dims, vocab_size)
    
    def forward(self, feature, captions):
        embedding = self.dropout(self.embedding(captions))
        embedding = torch.cat((feature.unsqueeze(1), embedding), dim=1)
        out , _ = self.rnn(embedding)
        out = self.fc(out)
        return out

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        embedding_dims,
        vocab_size,
        hidden_dims,
        num_layers,
        bidirectional,
        dropout
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = CNN(embedding_dims, dropout)
        self.decoder = RNN(
            vocab_size,
            embedding_dims,
            hidden_dims,
            num_layers,
            bidirectional,
            dropout
        )
    
    def forward(self, image, caption):
        features = self.encoder(image)
        output = self.decoder(features, caption)
        return output


if __name__ == "__main__":
    image = torch.randn(32, 3, 224, 224)
    model = CNN(100) 
    feature = model(image)
    caption = torch.randint(1, 100, (32, 100))
    model2 = RNN(100, 100, 50, 2)
    x = model2(feature, caption)
    full_model = EncoderDecoder(100, 100, 50, 2)
    x = full_model(image, caption)
    print(x.shape)
