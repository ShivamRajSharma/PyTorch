import torch 
import CONFIG
import numpy as np
import torch.nn as nn
import pickle 
import model_dispatcher
from PIL import Image
import albumentations as alb
from albumentations import pytorch as AT

max_len = 50

def predict(image_path):
    word_to_idx = pickle.load(open('../model/word_to_idx.pickle', 'rb'))
    idx_to_word = pickle.load(open('../model/idx_to_word.pickle', 'rb'))
    model = model_dispatcher.EncoderDecoder(
        embedding_dims=CONFIG.embedding_dims,
        vocab_size=len(word_to_idx),
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layer,
        dropout=CONFIG.dropout
    )


    model.load_state_dict(torch.load('../model/Image_Captioning.bin'))
    model.eval()
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = alb.Compose([
        alb.Normalize(mean, std, always_apply=True),
        alb.Resize(224, 224, always_apply=True),
        AT.ToTensor()
    ])
    image = np.array(Image.open(image_path).convert('RGB'))
    image = transform(image=image)['image']

    image = image[None, :]
    sentence = []
    with torch.no_grad():
        x = model.encoder(image).unsqueeze(1)

        state = None
        for _ in range(max_len):
            x, state = model.decoder.rnn(x, state)
            predict = model.decoder.fc(x.squeeze(0))
            # word = torch.softmax(predict, dim=-1)
            word = predict.argmax(-1)
            prediction = idx_to_word.get(word.item())
            if prediction == '<EOS>':
                break
            sentence.append(prediction)
            x = model.decoder.embedding(word.unsqueeze(1))
            print(x.shape)
    
    sentence = ' '.join(sentence)
    print(sentence)




if __name__ == "__main__":
    predict('horse.jpg')