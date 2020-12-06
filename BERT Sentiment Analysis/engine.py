import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm

def loss_fn(output, target):
    return nn.BCEWithLogitsLoss()(output, target)

def accuracy(output, target):
    output = (torch.sigmoid(output) > 0.5)*1
    acc = torch.mean((output == target)*1.0)
    return acc


def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0
    running_acc = 0
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        target = data['sentiment'].to(device)

        output = model(ids, mask, token_type_ids)

        loss = loss_fn(output, target.reshape(-1, 1))

        loss.backward()
        running_loss += loss.item()
        running_acc += accuracy(output, target.reshape(-1, 1)).item()

        optimizer.step()
        scheduler.step()

        for p in model.parameters():
            p.grad=None
    
    epoch_acc = (running_acc*100)/len(dataloader)
    epoch_loss = running_loss/len(dataloader)
    
    return [epoch_acc, epoch_loss]

def eval(model, dataloader, device):
    model.eval()
    running_acc = 0
    running_loss = 0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            target = data['sentiment'].to(device)

            output = model(ids, mask, token_type_ids)

            running_loss += loss_fn(output, target.reshape(-1, 1)).item()
            running_acc +=  accuracy(output, target.reshape(-1, 1)).item()
    epoch_loss = running_loss/len(dataloader)
    epoch_acc = running_acc*100/len(dataloader)
    
    return [epoch_acc, epoch_loss]
