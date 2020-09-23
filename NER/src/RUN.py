import CONFIG
import DataLoader 
import engine
import model 

import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run():
    df = pd.read_csv('../input/...').sample(frac=1).reset_index(drop=True)
    pos_lb = LabelEncoder().fit(df.pos.values)
    ner_lb = LabelEncoder().fit(df.ner.values)
    df.pos = pos_lb.transform(df.pos.values)
    df.ner = ner_lb.transform(df.ner.values)


    data = DataLoader.DataLoader(df)

    data_len = len(df)
    indices = np.arange(data_len)
    valid_len = (data_len*CONFIG.Valid_split)

    train_index = indices[valid_len:]
    valid_index = indices[:valid_len]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_index)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_index)

    pickle.dump(data.vocab.word_to_idx, open('word_to_idx.pickle', 'wb'))
    pickle.dump(pos_lb, open('pos_lb.pickle', 'wb'))
    pickle.dump(ner_lb, open('ner_lb.pickle', 'wb'))

    pad_idx = data.vocab.word_to_idx['<PAD>']

    train_loader = torch.utils.data.DataLoader(
        data,
        num_workers=4,
        batch_size=CONFIG.BATCH_SIZE,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx),
        sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        data,
        num_workers=4,
        batch_size=CONFIG.BATCH_SIZE,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx),
        sampler=valid_sampler
    )

    vocab_size = len(data.vocab.word_to_index)

    num_pos_class = df['pos'].nunqiue
    num_ner_class = df['ner'].nunique

    model = model.LSTMModel(
        vocab_size=vocab_size,
        embed_dims=CONFIG.EMBED_DIMS,
        hidden_dims=CONFIG.HIDDEN_DIMS,
        num_layers=CONFIG.NUM_HIDDEN_LAYER,
        dropout=CONFIG.DROPOUT,
        num_pos_class=num_pos_class,
        num_ner_class=num_ner_class
    )

    if torch.cuda.is_available():
        compute = 'cuda'
        torch.backends.cudnn.benchmark=True
    else:
        compute = 'cpu'
    
    device = torch.device(compute)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="min"
    )

    bes_loss = 1e32

    print('-----------[INFO] STARTING TRAINING ----------------')
    for epoch in range(CONFIG.EPOCHS):
        train_loss = engine.train(model, train_loader, optimizer, scheduler, device)
        eval_loss = engine.eval(model, valid_loader, device)
        print(f'EPOCH {epoch}/{CONFIG.EPOCHS}')
        print(f'TRAIN LOSS = {train_loss}')
        print(f'VAL LOSS = {val_loss}')
        if best_loss > eval_loss:
            best_loss = eval_loss
            best_model = model.state_dict()

    torch.save(best_model, CONFIG.Model_Path)
