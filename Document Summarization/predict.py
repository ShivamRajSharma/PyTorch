import CONFIG
import Transformer

import pickle
import spacy
import numpy as np 
import torch
import torch.nn as nn

def predict(sentence):
    max_len=20

    tokenizer = spacy.load('en_core_web_sm')
    device = torch.device('cpu')
    
    word_to_idx = pickle.load(open(CONFIG.INPUT_PATH + 'word_to_idx.pickle', 'rb'))
    idx_to_word = pickle.load(open(CONFIG.INPUT_PATH + 'idx_to_word.pickle', 'rb'))

    vocab_size = len(word_to_idx)
    pad_idx = word_to_idx["<PAD>"]

    model = Transformer.Transformer(
        input_vocab_size=vocab_size,
        out_vocab_size=vocab_size,
        max_len=CONFIG.max_len,
        embed_dims=CONFIG.embed_dims,
        pad_idx=pad_idx,
        heads=CONFIG.heads,
        forward_expansion=CONFIG.forward_expansion,
        num_layers=CONFIG.num_layers,
        dropout=CONFIG.dropout,
        device=device
    )

    model.load_state_dict(torch.load(CONFIG.MODEL_PATH))

    sentence_idx = []
    for word in tokenizer(sentence):
        word = str(word.text.lower())
        if word in word_to_idx:
            sentence_idx.append(word_to_idx[word])
        else:
            sentence_idx.append(word_to_idx["<UNK>"])
    
    summary_idx_ = [word_to_idx["<SOS>"]]
    sentence_idx = torch.tensor(sentence_idx).unsqueeze(0)
    

    summary = []

    print('predicting')
    for _ in range(max_len):
        summary_idx = torch.tensor(summary_idx_).unsqueeze(0)
        output = model(sentence_idx, summary_idx)
        output = torch.softmax(output, dim=-1)
        output = torch.argmax(output, dim=-1)[:, -1].item()
        summary_idx_.append(output)
        if output == word_to_idx["<EOS>"]:
            break
        summary.append(idx_to_word[output])
    
    summary = ' '.join(summary)
    print(summary)


if __name__ == "__main__":
    sentence = ""
    predict(sentence)


        




    

