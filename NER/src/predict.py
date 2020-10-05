import warnings
warnings.filterwarnings('ignore')

import CONFIG
import NERModel

import pickle
import spacy 
import torch 
import numpy as np
import pandas as pd

def predict(sentence):

    word_to_idx = pickle.load(open('../input/word_to_idx.pickle', 'rb'))
    pos_lb = pickle.load(open('../input/pos_lb.pickle', 'rb'))
    tag_lb = pickle.load(open('../input/tag_lb.pickle', 'rb'))

    num_pos_class = len(list(pos_lb.classes_))
    num_tag_class = len(list(tag_lb.classes_))

    model = NERModel.LSTMModel(
        vocab_size=len(word_to_idx),
        embed_dims=CONFIG.EMBED_DIMS,
        hidden_dims=CONFIG.HIDDEN_DIMS,
        num_layers=CONFIG.NUM_HIDDEN_LAYER,
        dropout=CONFIG.DROPOUT,
        bidirectional=CONFIG.BIDIRECTIONAL,
        num_pos_class=num_pos_class,
        num_tag_class=num_tag_class
    )
    model.load_state_dict(torch.load(CONFIG.Model_Path))

    sentence_idx = []
    tokenized_sentence = []
    tokenizer = spacy.load('en_core_web_sm')
    for word in tokenizer(sentence):
        word = str(word.text.lower())
        tokenized_sentence.append(word)
        if word in word_to_idx:
            sentence_idx.append(word_to_idx[word])
        else:
            sentence_idx.append(word_to_idx['<UNK>'])
    
    sentence_idx = torch.tensor(sentence_idx, dtype=torch.long).unsqueeze(0)
    
    sentence_idx = sentence_idx

    pos , tag = model(sentence_idx)

    pos =  pos.squeeze(0).argmax(1)
    tag = tag.squeeze(0).argmax(1)

    pos = pos.detach().cpu().numpy()
    tag = tag.detach().cpu().numpy()

    pos_out = pos_lb.inverse_transform(pos)
    tag_out = tag_lb.inverse_transform(tag)
    
    print(f'\nSENTENCE -> {sentence}\n')
    
    df = pd.DataFrame([tokenized_sentence, pos_out, tag_out]).transpose()
    df.columns = ['WORD', 'POS', 'TAG']
    print(df)

if __name__ == '__main__':
    sentence = str(input('ENTER A SENTENCE -> '))
    predict(sentence)
