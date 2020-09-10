import torch 
import torch.nn as nn
import numpy as np 
from tqdm import tqdm 

def loss_fn(output, target, pad_index):
    return nn.CrossEntropyLoss(ignore_index=pad_index)(output, target)

def train(model, dataloader, device, optimizer, scheduler, pad_index):
    model.train()
    running_loss = 0
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        english_sentences = data['english_idx']
        german_sentences = data['german_idx']
        english_sentences = english_sentences.to(device)
        german_sentences = german_sentences.to(device)
        output = model(
            german_sentences,
            english_sentences,
            teacher_force_ratio=0.5
        )
        output = output[:, 1:, :].reshape(-1, output.shape[2]).to(device)
        english_sentences = english_sentences[:, 1:].reshape(-1).to(device)
        loss = loss_fn(output, english_sentences, pad_index)
        
        loss.backward()
        running_loss += loss.item()

        optimizer.step()
        scheduler.step()
        for p in model.parameters():
            p.grad=None

    epoch_loss = running_loss/len(dataloader)
    return epoch_loss


def eval_fn(model, dataloader, device, pad_index):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            english_sentences = data['english_idx']
            german_sentences = data['german_idx']
            english_sentences = english_sentences.to(device)
            german_sentences = german_sentences.to(device)
            output = model(
                german_sentences,
                english_sentences,
                teacher_force_ratio=1.0
            )
            output = output[:, 1:, :].reshape(-1, output.shape[2]).to(device)
            english_sentences = english_sentences[:, 1:].reshape(-1).to(device)
            loss = loss_fn(output, english_sentences, pad_index)
            running_loss += loss.item()

        epoch_loss = running_loss/len(dataloader)
        return epoch_loss