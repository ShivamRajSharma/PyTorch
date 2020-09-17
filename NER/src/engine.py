import torch 
import torch.nn as nn
from tqdm import tqdm 

def loss_fn(ouput, target):
    return  nn.CrossEntropyLoss()(ouput, target)

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0
    
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None

        sentence = data['sentence'].to(device)
        sentence_idx = data['sentence_idx'].to(device)
        pos_tag = data['pos_tag'].to(device)
        ner_tag = data['ner_tag'].to(device)

        pos_output, ner_output = model(sentence_idx)
        
        loss_pos = loss_fn(pos_output.reshape(-1, pos_output.shape[-1]), pos_tag.reshape(-1, 1))
        loss_ner = loss_fn(ner_output.reshape(-1, ner_output.shape[-1]), ner_tag.reshape(-1, 1))

        loss = (loss_pos + loss_ner)/2

        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
    
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss



def eval(model, dataloader, device):
    running_loss = 0
    model.eval()  
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            sentence = data['sentence'].to(device)
            sentence_idx = data['sentence_idx'].to(device)
            pos_tag = data['pos_tag'].to(device)
            ner_tag = data['ner_tag'].to(device)

            pos_output, ner_output = model(sentence_idx)
            
            loss_pos = loss_fn(pos_output.reshape(-1, pos_output.shape[-1]), pos_tag.reshape(-1, 1))
            loss_ner = loss_fn(ner_output.reshape(-1, ner_output.shape[-1]), ner_tag.reshape(-1, 1))

            loss = (loss_pos + loss_ner)/2

            running_loss += loss.item()
    epoch_loss = running_loss/len(dataloader)

    return epoch_loss

        