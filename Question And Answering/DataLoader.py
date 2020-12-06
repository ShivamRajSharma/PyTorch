import CONFIG

import torch 
import torch.nn as nn
import numpy as np 
from torch.nn.utils.rnn import pad_sequence

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text = df.text.values
        self.sentiment = df.sentiment.values
        self.selected_text = df.selected_text.values
        self.max_len = CONFIG.Max_Len
        self.tokeinizer = CONFIG.tokenizers
        self.sentiment_map = self.mapping()
        
    def mapping(self):
        sentiment_dict = {}
        sentiment = ['positive', 'negative', 'neutral']
        for x in sentiment:
            inputs = self.tokeinizer.encode(x)
            sentiment_dict[x] = inputs.ids[0]
        return sentiment_dict

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = ' ' + ' '.join(self.text[idx].strip().split())
        sentiment =  self.sentiment[idx]
        selected_text = ' ' + ' '.join(self.selected_text[idx].strip().split())

        start = None
        end = None
        len_selected = len(selected_text) - 1
        for num in (i for i, e in enumerate(text) if e == selected_text[1]):
            if  ' ' + text[num:num + len_selected] == selected_text:
                start = num
                end = num + len_selected -1
                break
        
        char_targets = [0]*len(text)

        if start != None and end != None:
            for i in range(start, end + 1):
                char_targets[i] = 1
        
        inputs = self.tokeinizer.encode(text)
        ids = inputs.ids
        offsets = inputs.offsets

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        target_start_logits = target_idx[0]
        target_end_logits = target_idx[-1]

        input_ids = [0] + [self.sentiment_map[sentiment]] + [2] +[2] + ids + [2]

        token_type_ids = [0, 0, 0, 0] + [0]*(len(ids) + 1)
        mask =  [1]*len(token_type_ids)

        tweet_offsets = [(0, 0)]*4 + offsets + [(0, 0)]

        target_start_logits += 4
        target_end_logits += 4

        
        return {
            'ids' : torch.tensor(input_ids, dtype=torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
            'target_start_logits' : torch.tensor(target_start_logits, dtype=torch.long),
            'target_end_logits' : torch.tensor(target_end_logits, dtype=torch.long),
            'offsets' : tweet_offsets,
            'text' : text,
            'sentiment' : sentiment,
            'selected_text' : selected_text
        }


class MyCollate:
    def __init__(self):
        self.pad_idx = 1
        self.mask_ids = 1
        self.token_type_ids_idx = 1
    
    def __call__(self, batch):
        ids = [item['ids'] for item in batch]
        mask = [item['mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        target_start_logits = torch.cat([item['target_start_logits'].unsqueeze(0) for item in batch], dim=0)
        target_end_logits = torch.cat([item['target_end_logits'].unsqueeze(0) for item in batch], dim=0)
        offsets = [item['offsets'] for item in batch]
        text = [item['text'] for item in batch]
        sentiment = [item['sentiment'] for item in batch]
        selected_text = [item['selected_text'] for item in batch]

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_idx)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        max_len = len(ids[1])
        pad_offsets = []
        for offset in offsets:
            pad_len = max_len - len(offset)
            offset += [(0, 0)]*pad_len
            pad_offsets.append(offset)
        


        return {
            'ids' : ids,
            'mask' : mask,
            'token_type_ids' : token_type_ids,
            'target_start_logits' : target_start_logits,
            'target_end_logits' : target_end_logits,
            'offsets' : pad_offsets,
            'text' : text,
            'sentiment' : sentiment,
            'selected_text' : selected_text
        }