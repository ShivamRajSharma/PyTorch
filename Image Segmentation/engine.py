import CONFIG

import torch 
import torch.nn as nn 
from tqdm import tqdm

def dice_loss(predicted, target, epsilon):
    predicted = torch.sigmoid(predicted)
    predicted = predicted.view(-1)
    target = target.view(-1)
    num = 2*((predicted*target).sum()) + epsilon
    den = predicted.sum()**2 + target.sum()**2 + epsilon
    return num/den


def loss_fn(predicted, target):
    epsilon = 1
    dice_loss_ = dice_loss(predicted, target, epsilon)
    bce_loss = nn.BCEWithLogitsLoss()(predicted, target)
    return (CONFIG.bce_loss_coeff*bce_loss) - (CONFIG.dice_loss_coeff*dice_loss_)

def train_fn(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad = None
        original_image = data['original_image'].to(device)
        target_image = data['mask'].to(device)
        prediction = model(original_image)
        loss = loss_fn(prediction, target_image)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss


def eval_fn(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            original_image = data['original_image'].to(device)
            target_image = data['mask'].to(device)
            prediction = model(original_image)
            loss = loss_fn(prediction, target_image)
            running_loss += loss.item()
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss
