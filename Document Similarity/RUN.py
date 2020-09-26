import CONFIG
import DataLoader
import DocSimModel
import engine

import torch
import torch.nn as nn
import pandas as pd 
import numpy as np 
import os 

def run():
    df = pd.read_csv(CONFIG.input_path).sample(frac=1).reset_index(drop=True)[:100]
    print('------- [INFO] TOKENIZING -------')
    loader = DataLoader.DataLoader(df)
    
    split = int(0.1*len(loader))
    indices = list(range(len(loader)))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.RandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.RandomSampler(val_indices)

    pad_idx = loader.vocab.word_to_idx['<PAD>']

    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=CONFIG.Batch_Size,
        num_workers=2,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx),
        sampler=train_sampler
    )
    
    val_loader = torch.utils.data.DataLoader(
        loader,
        num_workers=2,
        batch_size=CONFIG.Batch_Size,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx),
        sampler=val_sampler
    )

    # if torch.cuda.is_available():
    # accelarator = 'cuda'
    #     torch.backends.cudnn.benchmark = True
    # else:
    #     accelarator = 'cpu'

    accelarator = 'cpu'
    
    device = torch.device(accelarator)

    model = DocSimModel.DocSimModel(
        voacb_size=len(loader.vocab.word_to_idx),
        embed_dims=CONFIG.embed_dims,
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layers,
        bidirectional=CONFIG.bidirectional,
        dropout=CONFIG.dropout,
        out_dims=CONFIG.out_dims
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    best_loss = 1e4
    
    scheduler = None

    print('------------------------------ [INFO] STARTING TRAINING --------------------------------')
    for epoch in range(CONFIG.Epochs):
        print(f'----- EPOCH - {epoch+1}/ {CONFIG.Epochs} -----')
        train_loss = engine.train_fn(model, train_loader, optimizer, scheduler, device)
        eval_loss = engine.eval_fn(model, val_loader, device)
        print(f'Train Loss = {train_loss} | Eval Loss = {eval_loss}\n')
        if best_loss > eval_loss:
            best_loss = eval_loss
            best_model = model.state_dict()
    
    torch.save(best_model, CONFIG.MODEL_PATH)

if __name__ == "__main__":
    run()