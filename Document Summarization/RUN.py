import CONFIG
import DataLoader
import engine
import predict
import Transformer

import torch 
import pickle
import transformers
import torch.nn as nn 
import pandas as pd 

import sys

def run():
    df = pd.read_csv(CONFIG.INPUT_PATH + 'news_summary_more.csv').sample(frac=CONFIG.frac).reset_index(drop=True)
    print('--------- [INFO] TOKENIZING --------')
    loader = DataLoader.DataLoader(df)
    print(f'len of loader = {len(loader)}')

    split = int(CONFIG.split*len(loader))
    indices = list(range(len(loader)))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.RandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.RandomSampler(val_indices)

    pickle.dump(loader.vocab.word_to_idx, open(CONFIG.INPUT_PATH + 'word_to_idx.pickle', 'wb'))
    pickle.dump(loader.vocab.idx_to_word, open(CONFIG.INPUT_PATH + 'idx_to_word.pickle', 'wb'))

    pad_idx = loader.vocab.word_to_idx["<PAD>"]

    train_loader = torch.utils.data.DataLoader(
        loader, 
        batch_size=CONFIG.Batch_Size,
        num_workers=4,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx),
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=CONFIG.Batch_Size,
        num_workers=4,
        pin_memory=True,
        collate_fn=DataLoader.MyCollate(pad_idx),
        sampler=val_sampler
    )

    # if torch.cuda.is_available():
    #     accelarator = 'cuda'
    #     torch.backends.cudnn.benchmark = True
    # else:
    #     accelarator = 'cpu'

    accelarator = 'cpu'
    
    vocab_size = len(loader.vocab.word_to_idx)
    
    device = torch.device(accelarator)

    model = Transformer.Transformer(
        input_vocab_size=vocab_size,
        out_vocab_size=vocab_size,
        max_len=CONFIG.max_len,
        embed_dims=CONFIG.embed_dims,
        pad_idx=pad_idx,
        heads=CONFIG.heads,
        forward_expansion=CONFIG.forward_expansion,
        num_layers=CONFIG.num_layers,
        dropout=CONFIG.dropout,
        device = device
    )

    model = model.to(device)
    decay_parmas = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimized_params = [
        {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in decay_parmas)], 'weight_decay': 0.001},
        {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in decay_parmas)], 'weight_decay': 0.0}
    ]


    optimizer = transformers.AdamW(optimized_params, lr=CONFIG.LR)
    num_training_steps = CONFIG.Epochs*len(loader)//CONFIG.Batch_Size
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG.Warmup_steps*num_training_steps,
        num_training_steps=num_training_steps
    )

    best_loss = 1e4
    best_model = model.state_dict()
    print('--------- [INFO] STARTING TRAINING ---------')
    for epoch in range(CONFIG.Epochs):
        train_loss = engine.train_fn(model, train_loader, optimizer, scheduler, device, pad_idx)
        val_loss = engine.eval_fn(model, val_loader, device, pad_idx)
        print(f'EPOCH -> {epoch+1}/{CONFIG.Epochs} | TRAIN LOSS = {train_loss} | VAL LOSS = {val_loss}')
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, CONFIG.MODEL_PATH)
            predict.predict('''Saurav Kant, an alumnus of upGrad and IIIT-B's PG Program in Machine learning and Artificial Intelligence, was a Sr Systems Engineer at Infosys with almost 5 years of work experience. The program and upGrad's 360-degree career support helped him transition to a Data Scientist at Tech Mahindra with 90% salary hike. upGrad's Online Power Learning has powered 3 lakh+ careers.''')

if __name__ == "__main__":
    run()