import torch
import torch.nn as nn
from tqdm import tqdm 

def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)


def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None

        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        start_logits_target = data['target_start_logits'].to(device)
        end_logits_target = data['target_end_logits'].to(device)
        
        start_logits, end_logits = model(ids, mask, token_type_ids)
        
        loss_start_logits = loss_fn(start_logits, start_logits_target)
        loss_end_logits = loss_fn(end_logits, end_logits_target)

        loss = loss_end_logits + loss_start_logits

        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
    
    epoch_loss = running_loss / len(dataloader)
    
    return epoch_loss


def eval(model, dataloader, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            start_logits_target = data['target_start_logits'].to(device)
            end_logits_target = data['target_end_logits'].to(device)
            
            start_logits, end_logits = model(ids, mask, token_type_ids)

            loss_start_logits = loss_fn(start_logits, start_logits_target)
            loss_end_logits = loss_fn(end_logits, end_logits_target)

            loss = loss_end_logits + loss_start_logits

            running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataloader)

    return epoch_loss, jaccard_scoreq