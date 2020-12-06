import engine
import DataLoader
import MODEL
import CONFIG

import os
import sys
import glob
import torch 
import pickle
import torchaudio
import numpy as  np 
import pandas as pd 
import torch.nn as nn
from sklearn.model_selection import train_test_split

def data_preprocess(path):
    path_and_label = []
    files = os.listdir(path)
    n = 0
    for file_ in files:
        file_path = os.path.join(path, file_)
        sub_files = os.listdir(file_path)
        for sub_file in sub_files:
            sub_file_path = os.path.join(file_path, sub_file)
            main_text_file = glob.glob(sub_file_path  + '/*.txt')
            f = open(main_text_file[0]).read().strip().split('\n')
            for line in f:
                audio_name = line.split()[0]
                audio_file_full_path = os.path.join(sub_file_path, audio_name + '.flac')
                label = ' '.join(line.split()[1:])
                label = label.lower()
                path_and_label.append([audio_file_full_path, label])
                n+=1 
                if not os.path.exists(audio_file_full_path):
                    print(audio_file_full_path)
    return path_and_label
            

def run():
    dataset_path = os.path.join(CONFIG.dataset_path, 'LibriSpeech/train-clean-360')
    path_and_label = data_preprocess(dataset_path)
    train_data, val_data  = train_test_split(path_and_label, test_size=0.1)

    train_data.sort(key=lambda x : len(x[1]))
    val_data.sort(key=lambda x : len(x[1]))
    
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    char_to_idx = pickle.load(open('../input/char_to_idx.pickle', 'rb'))

    transforms = [
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    ]

    
    train_data = DataLoader.DataLoader(train_data, char_to_idx, transforms)
    val_data = DataLoader.DataLoader(val_data, char_to_idx)


    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=CONFIG.Batch_Size,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx=0)
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=CONFIG.Batch_Size,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx=0)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=True
    
    model = MODEL.ASRModel(
        input_channel=1, 
        out_channel=CONFIG.out_channel, 
        kernel_size=CONFIG.kernel_size, 
        padding=CONFIG.padding,
        num_inception_block=CONFIG.num_inception_block,
        squeeze_dims=CONFIG.squeeze_dims,
        rnn_input_dims=CONFIG.rnn_input_dims,
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layers,
        bidirectional=CONFIG.bidirectional,
        dropout=CONFIG.dropout,
        num_classes=CONFIG.num_classes
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=CONFIG.scheduler_threshold,
        mode='min',
        patience=CONFIG.scheduler_patience,
    )

    best_loss = 1e4

    print('-------------- [INFO] STARTING TRAINING ---------------')
    for epoch in range(CONFIG.Epochs):
        train_loss = engine.train_fn(model, train_loader, scheduler, device)
        val_loss = engine.eval_fn(model, val_loader, device)
        scheduler.step(val_loss)
        print(f'\nEPOCH -> {epoch+1}/{CONFIG.Epochs} | TRAIN LOSS -> {train_loss} | VAL LOSS -> {val_loss} \n')
        if best_loss > val_loss:
            best_loss=val_loss
            best_model = model.state_dict()
            torch.save(best_model, CONFIG.MODEL_PATH)



if __name__ == "__main__":
    run()
