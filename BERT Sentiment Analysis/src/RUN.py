import DataLoader
import model_dispatcher
import engine
import CONFIG

import torch
import torch.nn as nn
import pandas as pd 
import numpy as np 
import transformers 
from sklearn.model_selection import train_test_split

import sys

def run():
    df = pd.read_csv('input/IMDB Dataset.csv').sample(frac=1).reset_index(drop=True)
    df.sentiment = df.sentiment.apply(
        lambda x:
        1 if x == 'positive' else 0
    )
    df_train, df_val = train_test_split(df, test_size=0.1)

    train_data = DataLoader.DataLoader(df_train)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=CONFIG.Batch_Size,
        num_workers=4,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate()
    )

    val_data = DataLoader.DataLoader(df_val)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=CONFIG.Batch_Size,
        num_workers=4,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate()
    )
    

    if torch.cuda.is_available():
        compute = 'cuda'
        torch.backends.cudnn.benchmark=True
    else:
        compute = 'cpu'
    
    device = torch.device(compute)


    model = model_dispatcher.BERTMODEL()

    model = model.to(device)

    model_param = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    
    optimizer_param = [
        {'params': [p for n, p in model_param if not any(nd in n for nd in no_decay)], 'weight_decay':0.001},
        {'params': [p for n, p in model_param if any(nd in n for nd in no_decay)], 'weight_decay':0.0 }
    ]

    num_training_steps = len(df_train)*CONFIG.Epochs//CONFIG.Batch_Size

    optimizer = transformers.AdamW(optimizer_param, lr=3e-5)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_loss = 1e32
    best_model = None
    print('---------- [INFO] STARTING TRAINING ---------')
    for epoch in range(CONFIG.Epochs):
        train_acc, train_loss = engine.train(
            model, 
            train_loader, 
            optimizer, 
            scheduler, 
            device
        )

        val_acc, val_loss = engine.eval(
            model,
            val_loader,
            device
        )

        print(f'EPOCH : {epoch+1}/{CONFIG.Epochs}')
        print(f'TRIAN_ACC = {train_acc} | TRAIN LOSS = {train_loss}')
        print(f'VAL ACC = {val_acc} | VAL LOSS {val_loss}')

        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
    torch.save(best_model, CONFIG.Model_Path)

    


if __name__ == "__main__":
    run()
