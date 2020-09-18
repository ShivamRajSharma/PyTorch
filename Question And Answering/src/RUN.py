import CONFIG
import DataLoader
import engine
import ROBERTA_MODEL

import numpy as np 
import torch 
import transformers
import torch.nn as nn
import pandas as pd 
from sklearn.model_selection import train_test_split

def run():
    df = pd.read_csv('input/train.csv').sample(frac=1).dropna().reset_index(drop=True)[:30]

    df_train, df_val = train_test_split(df, test_size=0.1, stratify=df.sentiment.values)


    df_train['len'] = df_train['text'].apply(lambda x : len(x))
    df_train.sort_values(by='len', inplace=True)
    df_train.drop(['len'], axis=1, inplace=True)

    df_val['len'] = df_val['text'].apply(lambda x: len(x))
    df_val.sort_values(by='len', inplace=True)
    df_val.drop(['len'], axis=1, inplace=True)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    train_data = DataLoader.DataLoader(df_train)
    val_data = DataLoader.DataLoader(df_val)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=0,
        batch_size=CONFIG.Batch_Size,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate()
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=0,
        batch_size=CONFIG.Batch_Size,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate()
    )

    torch.backends.cudnn.benchmark = True

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = ROBERTA_MODEL.QnAModel(CONFIG.Dropout)
    mode = model.to(device)

    parameters = list(model.parameters())

    decay_parameter = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimized_parameters = [
        {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in decay_parameter)], 'weight_decay': 0.001},
        {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd  in decay_parameter)], 'weight_decay': 0.0}
    ]

    optimizer = transformers.AdamW(optimized_parameters, lr=3e-4)

    num_training_steps = len(df_train)*CONFIG.Epochs/CONFIG.Batch_Size
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=0
    )

    best_loss = 1e32
    best_jaccard = -1e32
    print('-------------[INFO] STARTING TRANING--------------')
    for epoch in range(CONFIG.Epochs):
        train_loss = engine.train(model, train_loader, optimizer, scheduler, device)
        val_loss = engine.eval(model,  val_loader, device)
        # jaccard_score = engine.eval(, device)
        print(f'EPOCHS - {epoch+1}/{CONFIG.Epochs}')
        # print(f'TRAINING LOSS -> {train_loss} | VAL LOSS -> {val_loss} | JACCARD SCORE -> {jaccard_score}')
        print(f'TRAINING LOSS -> {train_loss} | VAL LOSS -> {val_loss}')

        if jaccard_score > best_jaccard:
            best_jaccard = jaccard_score
            best_model = model.state_dict()
        
    torch.save(best_model, CONFIG.MODLE_PATH)


if __name__ == '__main__':
    run()