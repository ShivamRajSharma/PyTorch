import torch 
import torch.nn as nn
from tqdm import tqdm 

def accuracy(output, target, device):
    output = torch.tensor(output).to(device)
    predicted = output.reshape(-1)
    target = target.reshape(-1)
    acc = torch.mean((predicted == target)*1.0)
    return acc

def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0
    running_pos_acc = 0
    
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        sentences_idx = data['sentence_idx'].to(device)
        poss = data['pos'].to(device)
        accumalated_loss = 0
        for sentence_idx, pos in zip(sentences_idx, poss):
            loss = model.training_fn(sentence_idx, pos)
            accumalated_loss += loss
            running_loss += loss.item()
        accumalated_loss.backward()
        optimizer.step()
            
    
    epoch_loss = running_loss/(len(dataloader)*len(data['pos']))
    
    return epoch_loss



def eval(model, dataloader, device):
    running_loss = 0
    running_pos_acc = 0
    model.eval()  
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            sentences_idx = data['sentence_idx'].to(device)
            poss = data['pos'].to(device)
            for sentence_idx, pos in zip(sentences_idx, poss):
                loss = model.training_fn(sentence_idx, pos)
                running_loss += loss.item()
                output = model(sentence_idx)
                acc = accuracy(output[1], pos, device)
                running_pos_acc += acc.item()

                
    epoch_loss = running_loss/(len(dataloader)*len(data['pos']))
    pos_acc = running_pos_acc/(len(dataloader)*len(data['pos']))

    return epoch_loss, pos_acc   
