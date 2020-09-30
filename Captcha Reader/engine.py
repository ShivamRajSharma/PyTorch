import torch 
import torch.nn as nn 
from tqdm import tqdm

def loss_fn(predicted, target, out_len, target_len, blank):
    return nn.CTCLoss(blank=blank)(predicted, target, out_len, target_len)

def train_fn(model, dataloader, optimizer, blank, device):
    model.train()
    running_loss = 0
    for num_steps, data in tqdm(enumerate(dataloader),total=len(dataloader)):
        for p in model.parameters():
            p.grad=None
        capt_image = data['image'].to(device)
        target  = data['target'].to(device)
        out_len = data['out_len'].to(device)
        target_len = data['target_len'].to(device)
        prediction = model(capt_image)
        prediction = prediction.permute(1, 0, 2)
        prediction = nn.functional.log_softmax(prediction, dim=2)
        loss = loss_fn(prediction, target, out_len, target_len, blank)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss


def eval_fn(model, dataloader, blank, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            capt_image = data['image'].to(device)
            target  = data['target'].to(device)
            out_len = data['out_len'].to(device)
            target_len = data['target_len'].to(device)
            prediction = model(capt_image)
            prediction = prediction.permute(1, 0, 2)
            prediction = nn.functional.log_softmax(prediction, dim=2)
            loss = loss_fn(prediction, target, out_len, target_len, blank)
            running_loss += loss.item()
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss