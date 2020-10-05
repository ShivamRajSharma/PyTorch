import warnings
warnings.filterwarnings('ignore')

import CONFIG
import DataLoader 
import engine
import NERModel 

import os
import sys
import pickle
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run():
    df = pd.read_csv('input/ner_dataset.csv', encoding='latin-1')
    df['Sentence #'] = df['Sentence #'].fillna(method='ffill')
    
    if  os.path.exists('input/pos_lb.pickle') and  os.path.exists('tag_lb.pickle'):
        pos_lb = pickle.load(open('input/pos_lb.pickle', 'rb'))
        tag_lb = pickle.load(open('input/tag_lb.pickle', 'rb'))
    else:
        pos_lb = LabelEncoder().fit(df.POS.values)
        tag_lb = LabelEncoder().fit(df.Tag.values)
        pickle.dump(pos_lb, open('input/pos_lb.pickle', 'wb'))
        pickle.dump(tag_lb, open('input/tag_lb.pickle', 'wb'))

    df['POS'] = pos_lb.transform(df.POS.values)
    df['Tag'] = tag_lb.transform(df.Tag.values)

    pos_pad_idx = pos_lb.transform(['.'])[0]
    tag_pad_idx = tag_lb.transform(['O'])[0]

    
    sentence = df.groupby('Sentence #')['Word'].apply(list).values
    pos = df.groupby('Sentence #')['POS'].apply(list).values
    tag = df.groupby('Sentence #')['Tag'].apply(list).values

    print('-------- [INFO] TOKENIZING --------\n')
    data = DataLoader.DataLoader(sentence, pos, tag)

    data_len = len(data)
    indices = np.arange(0, data_len)
    valid_len = int(data_len*CONFIG.Valid_split)

    train_index = indices[valid_len:]
    valid_index = indices[:valid_len]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_index)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_index)

    if not os.path.exists('input/word_to_idx.pickle'):
        pickle.dump(data.vocab.word_to_idx, open('input/word_to_idx.pickle', 'wb'))

    pad_idx = data.vocab.word_to_idx['<PAD>']

    train_loader = torch.utils.data.DataLoader(
        data,
        num_workers=1,
        batch_size=CONFIG.BATCH_SIZE,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx, tag_pad_idx, pos_pad_idx),
        sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        data,
        num_workers=1,
        batch_size=CONFIG.BATCH_SIZE,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx, tag_pad_idx, pos_pad_idx),
        sampler=valid_sampler
    )


    vocab_size = len(data.vocab.word_to_idx)

    num_pos_class = len(list(pos_lb.classes_))
    num_tag_class = len(list(tag_lb.classes_))


    model = NERModel.LSTMModel(
        vocab_size=vocab_size,
        embed_dims=CONFIG.EMBED_DIMS,
        hidden_dims=CONFIG.HIDDEN_DIMS,
        num_layers=CONFIG.NUM_HIDDEN_LAYER,
        dropout=CONFIG.DROPOUT,
        bidirectional=CONFIG.BIDIRECTIONAL,
        num_pos_class=num_pos_class,
        num_tag_class=num_tag_class
    )

    if torch.cuda.is_available():
        accelarator = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        accelarator = 'cpu'
    
    device = torch.device(accelarator)


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    best_loss = 1e32

    print('\n-----------[INFO] STARTING TRAINING ----------------\n')
    for epoch in range(CONFIG.EPOCHS):
        train_loss, train_pos_acc, train_tag_acc  = engine.train(model, train_loader, optimizer, device)
        eval_loss, val_pos_acc, val_tag_acc = engine.eval(model, valid_loader, device)
        print(f'EPOCH -> {epoch+1}/{CONFIG.EPOCHS}')
        print(f'TRAIN LOSS = {np.round(train_loss, 5)} | TRAIN POS ACC = {np.round(train_pos_acc*100, 5)}% | TRAIN TAG ACC = {np.round(train_tag_acc*100, 5)}%')
        print(f'VAL LOSS   = {np.round(eval_loss, 5)} | VAL POS ACC   = {np.round(val_pos_acc*100, 5)}% | VAL TAG ACC = {np.round(val_tag_acc*100, 5)}%')
        if best_loss > eval_loss:
            best_loss = eval_loss
            best_model = model.state_dict()

    torch.save(best_model, CONFIG.Model_Path)

if __name__ == "__main__":
    run()
