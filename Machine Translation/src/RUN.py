import CONFIG 
import engine
import DataLoader
import torch
import torch.nn as nn 
from model import LSTM
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import numpy as np
from sklearn.model_selection import train_test_split
import sys

def run():
    english = open(CONFIG.english_path).read().strip().split('\n')[:25000]
    german = open(CONFIG.german_path).read().strip().split('\n')[:25000]
    
    print('----- [INFO] TRAINING DATA PREPROCESSING -----')
    data = DataLoader.TranslationDataset(english, german)

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


    encoder = LSTM.Encoder(
        input_dims=len(data.german_preprocessing.word_to_idx),
        embedding_size=300,
        hidden_size=1024,
        num_hidden_layer=2,
        dropout_ratio=0.5
    )

    decoder = LSTM.Decoder(
        input_dims=len(data.english_preprocessing.word_to_idx),
        embedding_dims=300,
        hidden_size=1024,
        num_hidden_layer=2,
        dropout_ratio=0.5,
        output_size=len(data.english_preprocessing.word_to_idx),
    )

    if torch.cuda.is_available():
        compute = 'cuda'
        torch.backends.cudnn.benchmark=True
    else:
        compute = 'cpu'
    
    device = torch.device(compute)
    
    model = LSTM.Encoder_Decoder(encoder, decoder, len(data.english_preprocessing.word_to_idx))
    model = model.to(device)

    num_training_steps = int(len(english)*CONFIG.EPOCHS/CONFIG.BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = CONFIG.WARMUP_STEPS,
        num_training_steps=num_training_steps
    )
    # scheduler = None

    best_loss = 1e32

    print('------[INF0] STARTING TRAINING ------')
    for epoch in range(CONFIG.EPOCHS):
        print(f'------EPOCH - {epoch+1} / {CONFIG.EPOCHS} -------')
        train_loss = engine.train(model, train_loader, device, optimizer, scheduler, eng_pad_idx)
        val_loss = engine.eval_fn(model, val_loader, device, eng_pad_idx)
        print(f'TRAIN_LOSS = {train_loss} | EVAL LOSS = {val_loss}')
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
        
        if epoch%1==0:
            sentence = 'ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen.'
            idx_sentence = [data.german_preprocessing.word_to_idx['<SOS>']]
            idx_sentence += data.german_preprocessing.numericalize(sentence)
            idx_sentence.append(data.german_preprocessing.word_to_idx['<EOS>'])
            idx_sentence = torch.tensor(idx_sentence, dtype=torch.long)
            idx_sentence = idx_sentence.unsqueeze(0).to(device)

            with torch.no_grad():
                hidden, cell = model.encoder(idx_sentence)
            english_translation = [data.english_preprocessing.word_to_idx['<SOS>']]
            translation = []

            max_len = 50
            for _ in range(max_len):
                previous_word = torch.tensor([english_translation[-1]]).to(device)
                output, hidden, cell = model.decoder(previous_word, hidden, cell)
                output = nn.Softmax(dim=-1)(output)
                best_guess = output.argmax(1).item()
                english_translation.append(best_guess)
                best_word = data.english_preprocessing.idx_to_word[best_guess]
                if best_word == '<EOS>':
                    break
                translation.append(best_word)
            translation = ' '.join(translation)
            print(f'Ground Truth -> a boat with several men on it is being pulled ashore by a large team of horses.')
            print(f'Translated-sentence -> {translation}')


    torch.save(best_model, CONFIG.MODEL_PATH)


if __name__ == "__main__":
    run()
