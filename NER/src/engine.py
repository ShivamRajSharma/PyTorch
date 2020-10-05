import torch 
import torch.nn as nn
from tqdm import tqdm 

def accuracy(output, target):
    predicted = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
    predicted = predicted.reshape(-1)
    target = target.reshape(-1)
    acc = torch.mean((predicted == target)*1.0)
    return acc


def loss_fn(output, target):
    return  nn.CrossEntropyLoss()(output, target)

def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0
    running_pos_acc = 0
    running_tag_acc = 0
    
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        sentence_idx = data['sentence_idx'].to(device)
        pos = data['pos'].to(device)
        tag = data['tag'].to(device)

        pos_output, tag_output = model(sentence_idx)
        
        pos_loss = loss_fn(pos_output.reshape(-1, pos_output.shape[-1]), pos.reshape(-1))
        tag_loss = loss_fn(tag_output.reshape(-1, tag_output.shape[-1]), tag.reshape(-1))

        running_pos_acc += accuracy(pos_output, pos).item()
        running_tag_acc += accuracy(tag_output, tag).item()


        loss = (pos_loss + tag_loss)/2
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    epoch_loss = running_loss/len(dataloader)
    pos_acc = running_pos_acc/len(dataloader)
    tag_acc = running_tag_acc/len(dataloader)
    
    return epoch_loss, pos_acc, tag_acc



def eval(model, dataloader, device):
    running_loss = 0
    running_pos_acc = 0
    running_tag_acc = 0
    model.eval()  
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            sentence_idx = data['sentence_idx'].to(device)
            pos = data['pos'].to(device)
            tag = data['tag'].to(device)

            pos_output, tag_output = model(sentence_idx)
            
            pos_loss = loss_fn(pos_output.reshape(-1, pos_output.shape[-1]), pos.reshape(-1))
            tag_loss = loss_fn(tag_output.reshape(-1, tag_output.shape[-1]), tag.reshape(-1))

            running_pos_acc += accuracy(pos_output, pos).item()
            running_tag_acc += accuracy(tag_output, tag).item()

            loss = (pos_loss + tag_loss)/2

            running_loss += loss.item()
    epoch_loss = running_loss/len(dataloader)
    pos_acc = running_pos_acc/len(dataloader)
    tag_acc = running_tag_acc/len(dataloader)

    return epoch_loss, pos_acc, tag_acc

        
