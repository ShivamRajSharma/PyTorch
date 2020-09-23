from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import DataLoader
import model_dispatcher
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


def run():
    df = pd.read_csv(CONFIG.Path)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    aug = albumentations.Compose([
        albumentations.Normalize(mean, std, always_apply=True),
        # albumentations.RandomCrop(200, 200),
        # albumentations.Blur(),
        # albumentations.RandomBrightness(),
        # albumentations.HueSaturationValue(),
        albumentations.Resize(224, 224, always_apply=True),
        # albumentations.RGBShift(),
        # albumentations.ChannelShuffle(),
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

    pickle.dump(dataset.vocab.word_to_idx, open('../model/word_to_idx.pickle','wb'))
    pickle.dump(dataset.vocab.idx_to_word, open('../model/idx_to_word.pickle', 'wb'))

    
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
    

    if torch.cuda.is_available():
        compute = 'cuda'
        torch.backends.cudnn.benchmark=True
    else:
        compute = 'cpu'
    
    device = torch.device(compute)

    model = model_dispatcher.EncoderDecoder(
        embedding_dims=CONFIG.embedding_dims,
        vocab_size=len(dataset.vocab.word_to_idx),
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layer,
        dropout=CONFIG.dropout
    )
    
    model = model.to(device)

    for name, param in model.encoder.base_model.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # num_training_steps = int(dataset_size*CONFIG.EPOCHS/CONFIG.BATCH_SIZE)
    
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_training_steps=num_training_steps,
    #     num_warmup_steps=0
    # )

    scheduler = None

    best_loss = 1e32
    
    print('------------------------------ [INFO] STARTING TRAINING --------------------------------')
    for epoch in range(CONFIG.EPOCHS):
        print(f'-----EPOCH - {epoch+1}/ {CONFIG.EPOCHS} -----')
        train_loss = engine.train(model, train_loader, optimizer, scheduler, device, pad_idx)
        eval_loss = engine.eval(model, val_loader, device, pad_idx)
        print(f'Train Loss = {train_loss} | Eval Loss = {eval_loss}\n')
        if best_loss > eval_loss:
            best_loss = eval_loss
            best_model = model.state_dict()
    
    torch.save(best_model, CONFIG.MODEL_PATH)

if __name__ == "__main__":
    run()
