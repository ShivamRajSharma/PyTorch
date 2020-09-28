import torch
import torch.nn as nn 
from tqdm import tqdm

def loss_fn(predicted, target, pad_idx):
    return nn.CrossEntropyLoss(ignore_index=pad_idx)(predicted, target)

def train_fn(model, dataloader, optimizer, scheduler, device, pad_idx):
    model.train()
    running_loss = 0
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        
        text =  data["text_idx"].to(device)
        headlines = data["headline_idx"].to(device)
        predicted = model(text, headlines[:, :-1])
        predicted = predicted.view(-1, predicted.shape[-1])
        headlines = headlines[:, 1:].reshape(-1)
        loss = loss_fn(predicted, headlines, pad_idx)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_loss = running_loss/len(dataloader)
    return epoch_loss


def eval_fn(model, dataloader, device, pad_idx):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            text =  data["text_idx"].to(device)
            headlines = data["headline_idx"].to(device)
            predicted = model(text, headlines[:, :-1])
            predicted = predicted.view(-1, predicted.shape[-1])
            headlines = headlines[:, 1:].reshape(-1)
            loss = loss_fn(predicted, headlines, pad_idx)
            running_loss += loss.item()
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss



