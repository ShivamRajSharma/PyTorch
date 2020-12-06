import warnings
warnings.filterwarnings('ignore')

import CONFIG
import DataLoader
import DocSimModel
import engine

import transformers
import torch
import pickle
import torch.nn as nn
import pandas as pd 
import numpy as np 
import os 
import sys

def run():
    df = pd.read_csv(CONFIG.input_path).sample(frac=1).reset_index(drop=True).fillna("")
    print('------- [INFO] TOKENIZING -------\n')
    
    if not os.path.exists('input/word_to_idx.pickle') or not os.path.exists('input/idx_to_word.pickle'):
        pickle.dump(loader.vocab.word_to_idx, open('input/word_to_idx.pickle', 'wb'))
        pickle.dump(loader.vocab.idx_to_word, open('input/idx_to_word.pickle', 'wb'))
    
    train_data = df[df['is_duplicate']==1]

    val_data = df[:10000]


    train_data = DataLoader.DataLoader(train_data)
    val_data = DataLoader.DataLoader(val_data)

    pad_idx = train_data.vocab.word_to_idx['<PAD>']

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=CONFIG.Batch_Size,
        num_workers=2,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=2,
        batch_size=CONFIG.Batch_Size,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx)
    )

    if torch.cuda.is_available():
        accelarator = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        accelarator = 'cpu'
    
    device = torch.device(accelarator)

    model = DocSimModel.DocSimModel(
        voacb_size=len(train_data.vocab.word_to_idx),
        embed_dims=CONFIG.embed_dims,
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layers,
        bidirectional=CONFIG.bidirectional,
        dropout=CONFIG.dropout,
        out_dims=CONFIG.out_dims
    )

    model = model.to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=CONFIG.LR, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=CONFIG.scheduler_threshold,
        mode='min',
        patience=CONFIG.scheduler_patience,
        factor=CONFIG.scheduler_decay_factor
    )

    if os.path.exists(CONFIG.CHECKPOINT):
        checkpoint = torch.load(CONFIG.CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpointed_epoch = checkpoint['epoch']
        print(f'\n-------------- [INFO] LOADING CHECKPOINT | EPOCH -> {checkpoint["epoch"]} | LOSS = {checkpoint["loss"]}--------')
    else:
        checkpointed_epoch = 0


    best_auc_roc = -1e4
    print('\n------------------------------ [INFO] STARTING TRAINING --------------------------------\n')
    for epoch in range(checkpointed_epoch, CONFIG.Epochs):
        train_loss = engine.train_fn(model, train_loader, optimizer, scheduler, device)
        val_auc_roc, val_loss = engine.eval_fn(model, val_loader, device)
        print(f'EPOCH -> {epoch+1}/ {CONFIG.Epochs} | TRAIN LOSS = {train_loss} | VAL AUC SCORE = {val_auc_roc} | VAL LOSS = {val_loss} | LR = {optimizer.param_groups[0]["lr"]}\n')
        scheduler.step(val_auc_roc)
        if best_auc_roc < val_auc_roc:
            best_auc_roc = val_auc_roc
            best_model = model.state_dict()
            torch.save(best_model, CONFIG.MODEL_PATH)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'auc_roc' : val_auc_roc
        }, CONFIG.CHECKPOINT)

if __name__ == "__main__":
    run()
