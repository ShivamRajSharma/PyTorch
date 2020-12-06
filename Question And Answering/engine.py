import CONFIG

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 


def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def calculate_jaccard_score(
        original_tweet, 
        target_string, 
        sentiment_val, 
        idx_start, 
        idx_end, 
        offsets,
        verbose=False
):

    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


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
    jaccard = []
    with torch.no_grad():
        for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            start_logits_target = data['target_start_logits'].to(device)
            end_logits_target = data['target_end_logits'].to(device)
            orig_selected_text = data['selected_text']
            offsets = data['offsets']
            sentiment = data['sentiment']
            orig_tweet = data['text']
            
            start_logits, end_logits = model(ids, mask, token_type_ids)

            loss_start_logits = loss_fn(start_logits, start_logits_target)
            loss_end_logits = loss_fn(end_logits, end_logits_target)

            start_logits = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
            end_logits = torch.softmax(end_logits, dim=1).detach().cpu().numpy()
            for num, tweet in enumerate(orig_tweet):
                selected_text = orig_selected_text[num]
                tweet_sentiment = sentiment[num]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet, 
                    target_string=selected_text, 
                    sentiment_val=tweet_sentiment, 
                    idx_start=np.argmax(start_logits[num, :]), 
                    idx_end=np.argmax(end_logits[num, :]), 
                    offsets=offsets[num]
                )
                jaccard.append(jaccard_score)

            loss = loss_end_logits + loss_start_logits

            running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_jaccard = np.mean(jaccard)

    return epoch_loss, epoch_jaccard