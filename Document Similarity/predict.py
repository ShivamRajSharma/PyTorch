import warnings
warnings.filterwarnings('ignore')

import CONFIG
import DocSimModel

import torch 
import spacy
import torch.nn as nn 
import pickle 
import numpy as np 

def predict(que_1, que_2, target):
    word_to_idx = pickle.load(open('input/word_to_idx.pickle', 'rb'))
    tokenizer = spacy.load('en_core_web_sm')
    model = DocSimModel.DocSimModel(
        voacb_size=len(word_to_idx),
        embed_dims=CONFIG.embed_dims,
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layers,
        bidirectional=CONFIG.bidirectional,
        dropout=CONFIG.dropout,
        out_dims=CONFIG.out_dims
    )
    trained_model = torch.load(CONFIG.CHECKPOINT)
    model.load_state_dict(torch.load(CONFIG.MODEL_PATH))
    model.eval()
    que_1_idx = []
    que_2_idx = []
    ques = [que_1, que_2]
    
    for num, que in enumerate(ques):
        for word in tokenizer(que):
            word = str(word.text.lower())
            if word in word_to_idx:
                idx = word_to_idx[word]
            else:
                idx = word_to_idx["<UNK>"]
            if num == 0:
                que_1_idx.append(idx)
            else:
                que_2_idx.append(idx)
        

    pad_idx = word_to_idx["<PAD>"]
    if len(que_1_idx) > len(que_2_idx):
        pad_len = len(que_1_idx) - len(que_2_idx)
        que_2_idx += [pad_idx]*pad_len
    else:
        pad_len = len(que_2_idx) - len(que_1_idx)
        que_1_idx += [pad_idx]*pad_len

    que_1_idx = torch.tensor([que_1_idx])
    que_2_idx = torch.tensor([que_2_idx])
    with torch.no_grad():
        que_1_vec, que_2_vec = model(que_1_idx, que_2_idx)

    similarity = nn.CosineSimilarity(dim=1)(que_1_vec, que_2_vec)[0].item()
    if similarity > CONFIG.similarity_thresh:
        result = 'Similar'
    else:
        result = 'Disimilar'
    
    target = 'Disimilar' if target == 0 else 'Similar'
    
    print(f'QUE 1) {que_1}')
    print(f'QUE 2) {que_2}')
    print(f'RESULT -> {result} | GROUND_TRUTH = {target} | SIMILARITY = {similarity}\n')


if __name__ == "__main__":
    question1 = "Do they enjoy eating the dessert?"
    question2 = "Do they like hiking in the desert?"
    predict(question1, question2, 0)
    question1 = "When will I see you?"
    question2 = "When can I see you again?"
    predict(question1, question2, 1)
