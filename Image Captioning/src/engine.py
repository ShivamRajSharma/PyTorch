import torch 
import torch.nn as nn
from tqdm import tqdm

def loss_fn(output, target, pad_idx):
    return nn.CrossEntropyLoss(ignore_index=pad_idx)(output, target)

def train(model, dataloader, optimizer, scheduler, device, pad_idx):
    running_loss = 0.0
    model.train()
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
        captions = data['Captions']
        images = data['Images']
        captions = captions.to(device)
        images = images.to(device)
        
        output = model(images, captions[:, :-1])

        output = output.reshape(-1, output.shape[-1])
        captions = captions.reshape(-1)

        loss = loss_fn(output, captions, pad_idx)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        for p in model.parameters():
            p.grad=None 
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss 


def eval(model, dataloader, device, pad_idx):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            captions = data['Captions']
            images = data['Images']
            captions = captions.to(device)
            images = images.to(device)
            output = model(images, captions[:, :-1])
            output = output.reshape(-1, output.shape[-1])
            captions = captions.reshape(-1)
            loss = loss_fn(output, captions,  pad_idx)
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
    return epoch_loss