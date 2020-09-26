from tqdm import tqdm
import torch 
import torch.nn as nn 

def triplet_loss_fn(que_1_vec, que_2_vec, margin=0.25):
    score = torch.matmul(que_1_vec, que_2_vec.transpose(0, 1))
    batch_len = len(score)
    positive = torch.diagonal(score)
    negetive_only = score - 2*torch.eye(batch_len)
    close_negetive = negetive_only.max(axis=1)[0]
    zero_on_digonal_score = score*(1-torch.eye(batch_len))
    mean_negetive = torch.mean(zero_on_digonal_score, axis=1)
    triplet_loss_1 = torch.where(
        (margin - positive + close_negetive)>0, 
        (margin - positive + close_negetive), 
        torch.zeros(positive.shape[-1])
    )

    triplet_loss_2 = torch.where(
        (margin - positive + mean_negetive)>0, 
        (margin - positive + mean_negetive), 
        torch.zeros(positive.shape[-1])
    )
    triplet_loss = torch.mean(triplet_loss_1 + triplet_loss_2)

    return triplet_loss


def train_fn(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0 
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad=None
        que_1 = data['que_1'].to(device)
        que_2 = data['que_2'].to(device)
        target = data['target'].to(device)
        que_1_vec, que_2_vec = model(que_1, que_2)
        loss = triplet_loss_fn(que_1_vec, que_2_vec)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # scheduler.atep()
    epoch_loss =  running_loss/len(dataloader)
    return epoch_loss

def eval_fn(model, dataloader, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            que_1 = data['que_1'].to(device)
            que_2 = data['que_2'].to(device)
            target = data['target'].to(device)
            que_1_vec, que_2_vec = model(que_1, que_2)
            loss = triplet_loss_fn(que_1_vec, que_2_vec)
            running_loss += loss.item()
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss
