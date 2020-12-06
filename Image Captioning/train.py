from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import DataLoader
import ImageCaptioningModel
import CONFIG 
import engine 

import sys
import torch 
import pickle
import numpy as np
import pandas as pd
import albumentations
from albumentations import pytorch as AT
import torch.nn as nn 
from sklearn.model_selection import train_test_split

import predict


def run():
    df = pd.read_csv(CONFIG.Path).sample(frac=1).reset_index(drop=True)[:10]
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose([
        albumentations.Normalize(mean, std, always_apply=True),
        albumentations.RandomBrightness(),
        albumentations.HueSaturationValue(),
        albumentations.Resize(224, 224, always_apply=True),
        AT.ToTensor()
    ])

    print('-------[INFO] TOKENIZING CAPTIONS -------')
    dataset = DataLoader.DataLoader(df, aug)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1* dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    pickle.dump(dataset.vocab.word_to_idx, open('model/word_to_idx.pickle','wb'))
    pickle.dump(dataset.vocab.idx_to_word, open('model/idx_to_word.pickle', 'wb'))

    
    pad_idx = dataset.vocab.word_to_idx['<PAD>']

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG.BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=8,
        collate_fn=DataLoader.MyCollate(pad_idx=pad_idx)
    )

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG.BATCH_SIZE,
        sampler=valid_sampler,
        pin_memory=True,
        num_workers=8,
        collate_fn=DataLoader.MyCollate(pad_idx=pad_idx)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    model = ImageCaptioningModel.EncoderDecoder(
        embedding_dims=CONFIG.embedding_dims,
        vocab_size=len(dataset.vocab.word_to_idx),
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layer,
        bidirectional=CONFIG.bidirectional,
        dropout=CONFIG.dropout
    )

    torch.backends.cudnn.benchmark = True
    
    model = model.to(device)

    for name, param in model.encoder.base_model.named_parameters():
        if "linear" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        threshold=0.001,
        mode='min'
    )

    best_loss = 1e4
    
    print('------------------------------ [INFO] STARTING TRAINING --------------------------------')
    for epoch in range(CONFIG.EPOCHS):
        print(f'-----EPOCH - {epoch+1}/ {CONFIG.EPOCHS} -----')
        train_loss = engine.train(model, train_loader, optimizer, device, pad_idx)
        val_loss = engine.eval(model, val_loader, device, pad_idx)
        scheduler.step(val_loss)
        print(f'Train Loss = {train_loss} | Eval Loss = {val_loss}\n')
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, CONFIG.MODEL_PATH)
            predict.predict('1.jpg')

if __name__ == "__main__":
    run()
