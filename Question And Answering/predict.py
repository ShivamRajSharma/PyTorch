import warnings 
warnings.filterwarnings('ignore')

import torch 
import torch.nn as nn 
import transformers 
import CONFIG 
import ROBERTA_MODEL

def predict(sentence, sentiment, real_answer):
    tokenizer = CONFIG.tokenizers
    sentiment_inp = tokenizer.encode(sentiment)
    sentiment_idx = sentiment_inp.ids
    sentence_ = ' ' + ' '.join(sentence.strip().split())
    sentence_inp = tokenizer.encode(sentence_)
    offsets = sentence_inp.offsets
    offsets = [(0, 0)]*4 + offsets + [(0, 0)]
    sentence_idx = sentence_inp.ids
    input_ids = [0] + sentiment_idx + [2] + [2] + sentence_idx + [2]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    mask = [1]*len(input_ids)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = [0]*len(input_ids)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    model = ROBERTA_MODEL.QnAModel(CONFIG.Dropout)
    model.load_state_dict(torch.load(CONFIG.MODEL_PATH))
    model.eval()
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask, token_type_ids)

    start_logits = torch.softmax(start_logits, dim=-1)[0]
    end_logits = torch.softmax(end_logits, dim=-1)[0]

    start_logits = start_logits.argmax().item()
    end_logits = end_logits.argmax().item()

    if end_logits < start_logits:
        end_logits = start_logits

    filtered_output  = ""
    for ix in range(start_logits, end_logits + 1):
        filtered_output += sentence[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if len(sentence.split()) < 2:
        filtered_output = sentence
    print(f'Comprehension | {sentence.strip()}')
    print(f'Que           | Text indicating {sentiment} sentiment in the above comprehension? \n \nPred Ans      | {filtered_output.strip()} \nReal Ans      | {real_answer} \n-----------------------\n')


if __name__ == "__main__":
    import pandas as pd 
    import numpy as np
    df = pd.read_csv('input/train.csv')
    print('\n')
    idxs = np.random.randint(0, len(df), 5)
    for idx in idxs:
        row = df.iloc[idx]
        predict(row['text'], row['sentiment'], row['selected_text'])

