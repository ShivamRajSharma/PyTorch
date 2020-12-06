import CONFIG
import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm 

def loss_fn(output, target, mel_len, target_len):
    return torch.nn.CTCLoss(blank=CONFIG.blank_idx)(output, target, mel_len, target_len)

def train_fn(model, dataloader, optmizer, device):
    model.train()
    running_loss = 0 
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        mel_spect = data['mel_spect'].to(device)
        target = data['target'].to(device)
        mel_len = data['mel_len'].to(device)
        target_len = data['target_len'].to(device)

        output = model(mel_spect)

        output = output.permute(1, 0, 2)
        output = nn.functional.log_softmax(output, dim=2)
        loss = loss_fn(output, target, mel_len, target_len)

        running_loss += loss.item()
        loss.backward()
        optmizer.step()

    epoch_loss = running_loss / len(dataloader)
    
    return epoch_loss


def eval_fn(model, dataloader, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            mel_spect = data['mel_spect'].to(device)
            target = data['target'].to(device)
            mel_len = data['mel_len'].to(device)
            target_len = data['target_len'].to(device)

            output = model(mel_spect)

            output = output.permute(1, 0, 2)
            output = nn.functional.log_softmax(output, dim=2)
            loss = loss_fn(output, target, mel_len, target_len)

            running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataloader)
    
    return epoch_loss

