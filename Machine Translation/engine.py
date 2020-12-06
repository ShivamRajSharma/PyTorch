import CONFIG

import torch 
import torch.nn as nn
import numpy as np 
from tqdm import tqdm 
from torchtext.data.metrics import bleu_score

def loss_fn(output, target, pad_index):
    return nn.CrossEntropyLoss(ignore_index=pad_index)(output, target)

def train(model, dataloader, device, optimizer, pad_index):
    model.train()
    running_loss = 0
    for num_steps, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad=None
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        english_sentences = data['english_idx']
        german_sentences = data['german_idx']
        english_sentences = english_sentences.to(device)
        german_sentences = german_sentences.to(device)
        output = model(
            german_sentences,
            english_sentences,
            teacher_force_ratio=CONFIG.teacher_force_ratio
        )
        output = output[:, 1:, :].reshape(-1, output.shape[2]).to(device)
        english_sentences = english_sentences[:, 1:].reshape(-1).to(device)
        loss = loss_fn(output, english_sentences, pad_index)
        
        loss.backward()
        running_loss += loss.item()

        optimizer.step()

    epoch_loss = running_loss/len(dataloader)
    return epoch_loss


def decode(sentence_idx, english_idx_to_word):
    output_sentence = []
    for idx in sentence_idx:
        word = english_idx_to_word[idx]
        if word == "<SOS>":
            continue
        if word == "<EOS>":
            break
        output_sentence.append(word)
    return output_sentence


def calculate_bleu_score(
    model, 
    dataloader, 
    german_word_to_idx, 
    english_idx_to_word,
    device
    ):
    model.eval()
    predicted_sentences = []
    target_sentences = []
    with torch.no_grad():
        for num, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            german_idx = d['german_idx'].to(device)
            english_idx = d['english_idx'].to(device)

            predicted_english_idx = model(
                german_idx,
                english_idx,
                teacher_force_ratio=10
            )
            english_idx = english_idx.detach().cpu().numpy()
            predicted_english_idx = torch.softmax(predicted_english_idx, dim=-1)
            predicted_english_idx = predicted_english_idx.argmax(-1)
            predicted_english_idx = predicted_english_idx.detach().cpu().numpy()

            for num in range(len(predicted_english_idx)):
                target_idx = english_idx[num]
                output = predicted_english_idx[num]

                predicted_sentence = decode(output, english_idx_to_word)
                predicted_sentences.append(predicted_sentence)
                
                target_sentence = decode(target_idx, english_idx_to_word)
                target_sentences.append([target_sentence])
    
    return bleu_score(predicted_sentences, target_sentences)
