import warnings
warnings.filterwarnings('ignore')

import CONFIG
import NERCRFModel

import pickle
import spacy 
import torch 
import numpy as np
import pandas as pd

def predict(sentence):

    word_to_idx = pickle.load(open('input/word_to_idx.pickle', 'rb'))
    pos_lb = pickle.load(open('input/pos_lb.pickle', 'rb'))
    device = torch.device('cpu')

    num_pos_class = len(list(pos_lb.classes_))
    tag_to_idx = {str(x): num for num, x in enumerate(list(pos_lb.classes_))}
    tag_to_idx['start_tag'] = num_pos_class
    tag_to_idx['stop_tag'] = num_pos_class + 1

    model = NERCRFModel.NER(
        vocab_size=len(word_to_idx),
        embed_dims=CONFIG.EMBED_DIMS,
        hidden_dims=CONFIG.HIDDEN_DIMS,
        num_layers=CONFIG.NUM_HIDDEN_LAYER,
        num_classes=len(tag_to_idx),
        dropout=CONFIG.DROPOUT,
        bidirectional=CONFIG.BIDIRECTIONAL,
        tag_to_idx=tag_to_idx,
        device=device
    )

    model.load_state_dict(torch.load(CONFIG.Model_Path))
    model.eval()

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

    with torch.no_grad():
        pos = model(sentence_idx)[1]

    pos_out = pos_lb.inverse_transform(pos)
    
    df_dict = {'Sentence': tokenized_sentence, 'Predicted POS': pos_out}
    df = pd.DataFrame(df_dict)
    print(df)

if __name__ == '__main__':
    sentence = str(input('ENTER A SENTENCE -> '))
    predict(sentence)
    print('\n-------------------------------------------')