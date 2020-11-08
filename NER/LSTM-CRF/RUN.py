import warnings
warnings.filterwarnings('ignore')

import CONFIG
import DataLoader 
import engine
import NERCRFModel

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
    
    if  os.path.exists('input/pos_lb.pickle'):
        pos_lb = pickle.load(open('input/pos_lb.pickle', 'rb'))
    else:
        pos_lb = LabelEncoder().fit(df.POS.values)
        pickle.dump(pos_lb, open('input/pos_lb.pickle', 'wb'))

    df['POS'] = pos_lb.transform(df.POS.values)

    pos_pad_idx = pos_lb.transform(['.'])[0]

    
    sentence = df.groupby('Sentence #')['Word'].apply(list).values
    pos = df.groupby('Sentence #')['POS'].apply(list).values

    print('-------- [INFO] TOKENIZING --------\n')
    data = DataLoader.DataLoader(sentence, pos)

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
        num_workers=4,
        batch_size=CONFIG.BATCH_SIZE,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx, pos_pad_idx),
        sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        data,
        num_workers=4,
        batch_size=CONFIG.BATCH_SIZE,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx, pos_pad_idx),
        sampler=valid_sampler
    )


    vocab_size = len(data.vocab.word_to_idx)

    num_pos_class = len(list(pos_lb.classes_))
    tag_to_idx = {str(x): num for num, x in enumerate(list(pos_lb.classes_))}
    tag_to_idx['start_tag'] = num_pos_class
    tag_to_idx['stop_tag'] = num_pos_class + 1

    if torch.cuda.is_available():
        accelarator = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        accelarator = 'cpu'
    
    device = torch.device(accelarator)

    model = NERCRFModel.NER(
        vocab_size=vocab_size,
        embed_dims=CONFIG.EMBED_DIMS,
        hidden_dims=CONFIG.HIDDEN_DIMS,
        num_layers=CONFIG.NUM_HIDDEN_LAYER,
        num_classes=len(tag_to_idx),
        dropout=CONFIG.DROPOUT,
        bidirectional=CONFIG.BIDIRECTIONAL,
        tag_to_idx=tag_to_idx,
        device=device
    )


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    best_loss = 1e32

    print('\n-----------[INFO] STARTING TRAINING ----------------\n')
    for epoch in range(CONFIG.EPOCHS):
        train_loss = engine.train(model, train_loader, optimizer, device)
        eval_loss, val_pos_acc = engine.eval(model, valid_loader, device)
        print(f'EPOCH -> {epoch+1}/{CONFIG.EPOCHS}')
        print(f'TRAIN LOSS = {np.round(train_loss, 5)}')
        print(f'VAL LOSS   = {np.round(eval_loss, 5)} | VAL POS ACC   = {np.round(val_pos_acc*100, 5)}%')
        if best_loss > eval_loss:
            best_loss = eval_loss
            best_model = model.state_dict()

    torch.save(best_model, CONFIG.Model_Path)

if __name__ == "__main__":
    run()