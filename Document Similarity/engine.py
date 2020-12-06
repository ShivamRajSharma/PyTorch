import CONFIG

from tqdm import tqdm
import torch 
import torch.nn as nn 
import numpy as np
from sklearn.metrics import roc_auc_score

def triplet_loss_fn(que_1_vec, que_2_vec, device, margin):
    score = torch.matmul(que_1_vec, que_2_vec.t())
    batch_len = que_1_vec.shape[0]
    positive = torch.diagonal(score)
    negetive_only = score - 2*torch.eye(batch_len).to(device)
    close_negetive = negetive_only.max(axis=1)[0]
    zero_on_digonal_score = score*(1-torch.eye(batch_len).to(device))
    mean_negetive = torch.sum(zero_on_digonal_score, axis=1)/ (batch_len-1)
    triplet_loss_1 = torch.where(
        (margin - positive + close_negetive)>0, 
        (margin - positive + close_negetive), 
        torch.zeros(1).to(device)
    )
    triplet_loss_2 = torch.where(
        (margin - positive + mean_negetive)>0, 
        (margin - positive + mean_negetive), 
        torch.zeros(1).to(device)
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
        loss = triplet_loss_fn(que_1_vec, que_2_vec, device, CONFIG.margin)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss =  running_loss/len(dataloader)
    return epoch_loss

def eval_fn(model, dataloader, device):
    model.eval()
    running_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            que_1 = data['que_1'].to(device)
            que_2 = data['que_2'].to(device)
            target = data['target'].to(device)
            que_1_vec, que_2_vec = model(que_1, que_2)
            loss = triplet_loss_fn(que_1_vec, que_2_vec, device, CONFIG.margin)
            running_loss += loss.item()
            similarity = torch.diagonal(torch.matmul(que_1_vec, que_2_vec.t()))
            result = similarity.detach().cpu().numpy()
            result = np.where(result>0, result, 0)
            target = target.detach().cpu().numpy()
            outputs.extend(result)
            targets.extend(target)

    epoch_loss =  running_loss/len(dataloader)
    roc = roc_auc_score(targets, outputs)
    return roc, epoch_loss

if __name__ == "__main__":
    device = torch.device('cpu')
    v1 = torch.tensor([[0.26726124, 0.53452248, 0.80178373],[0.5178918 , 0.57543534, 0.63297887]])
    v2 = torch.tensor([[ 0.26726124,  0.53452248,  0.80178373],[-0.5178918 , -0.57543534, -0.63297887]])

    print("Triplet Loss:", triplet_loss_fn(v2, v1, device, 0.25))
