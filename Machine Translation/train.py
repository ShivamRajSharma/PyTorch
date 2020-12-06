import warnings
warnings.filterwarnings('ignore')

import CONFIG 
import engine
import predict
import DataLoader
import TranslationModel

import os
import pickle
import torch
import torch.nn as nn 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def run():
    english = open(CONFIG.english_path).read().strip().split('\n')[:25000]
    german = open(CONFIG.german_path).read().strip().split('\n')[:25000]
    
    print('----- [INFO] TRAINING DATA PREPROCESSING -----')
    data = DataLoader.TranslationDataset(english, german)

    if  not (os.path.exists('input/english_word_to_idx.pickle') or os.path.exists('input/english_idx_to_word.pickle')):
        pickle.dump(data.english_preprocessing.word_to_idx, open('input/english_word_to_idx.pickle', 'wb'))
        pickle.dump(data.english_preprocessing.idx_to_word, open('input/english_idx_to_word.pickle', 'wb'))
    
    if  not (os.path.exists('input/german_word_to_idx.pickle') or os.path.exists('input/german_idx_to_word.pickle')):
        pickle.dump(data.german_preprocessing.word_to_idx, open('input/german_word_to_idx.pickle', 'wb'))
        pickle.dump(data.german_preprocessing.idx_to_word, open('input/german_idx_to_word.pickle', 'wb'))
        

    eng_pad_idx = data.english_preprocessing.word_to_idx['<PAD>']
    grm_pad_idx = data.german_preprocessing.word_to_idx['<PAD>']


    dataset_size = len(data)
    indices = np.arange(dataset_size)
    split = int(np.floor(0.1*dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)


    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(
            eng_pad_idx=eng_pad_idx,
            grm_pad_idx=grm_pad_idx
        ),
        sampler=train_sampler

    )

    val_loader = torch.utils.data.DataLoader(
        data,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(
            eng_pad_idx=eng_pad_idx,
            grm_pad_idx=grm_pad_idx
        ),
        sampler=val_indices
    )


    encoder = TranslationModel.Encoder(
        vocab_size=len(data.german_preprocessing.word_to_idx),
        embedding_size=CONFIG.encoder_embed_dims,
        hidden_size=CONFIG.encoder_hidden_dims,
        num_hidden_layer=CONFIG.encoder_num_layers,
        dropout_ratio=CONFIG.encoder_dropout
    )
    

    decoder = TranslationModel.Decoder(
        vocab_size=len(data.english_preprocessing.word_to_idx),
        embedding_dims=CONFIG.decoder_embed_dims,
        hidden_size=CONFIG.decoder_hidden_dims,
        num_hidden_layer=CONFIG.decoder_num_layers,
        dropout_ratio=CONFIG.decoder_dropout,
        output_size=len(data.english_preprocessing.word_to_idx),
    )


    device = torch.device('cuda')
    model = TranslationModel.Encoder_Decoder(encoder, decoder, len(data.english_preprocessing.word_to_idx))
    model = model.to(device)

    num_training_steps = int(len(english)*CONFIG.EPOCHS/CONFIG.BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=CONFIG.patience,
        threshold=CONFIG.scheduler_thresh,
        mode="min",
        factor=CONFIG.decay_factor
    )

    best_bleu_score = -1


    past = {'train_loss': [], 'bleu_score': []}

    print('\n-------[INF0] STARTING TRAINING -------')
    for epoch in range(CONFIG.EPOCHS):
        
        train_loss = engine.train(
            model=model, 
            dataloader=train_loader, 
            device=device, 
            optimizer=optimizer, 
            pad_index=eng_pad_idx
        )


        bleu_score = engine.calculate_bleu_score(
            model=model, 
            dataloader=val_loader, 
            german_word_to_idx=data.german_preprocessing.word_to_idx, 
            english_idx_to_word=data.english_preprocessing.idx_to_word,
            device=device
        )

            
        print(f'EPOCH -> {epoch+1}/{CONFIG.EPOCHS} | TRAIN_LOSS = {train_loss} | BLEU SCORE = {bleu_score} | LR = {optimizer.param_groups[0]["lr"]}\n')

        scheduler.step(bleu_score)
        past['bleu_score'].append(bleu_score)
        past['train_loss'].append(train_loss)
        pickle.dump(past, open('past_loss_bleu.pickle', 'wb'))
        
        
        if best_bleu_score < bleu_score:
            best_bleu_score = bleu_score
            best_model = model.state_dict()
            torch.save(best_model, CONFIG.MODEL_PATH)
            idx = np.random.randint(0, len(german))
            input_sentence = german[idx]
            output_sentence = english[idx] 
            predicted_sentence = predict.predict(input_sentence)
            print(f'GROUND_TRUTH -> {output_sentence.lower()}')
            print(f'PREDICTION   -> {predicted_sentence}\n')


if __name__ == "__main__":
    run()
