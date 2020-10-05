import torch 
import torch.nn as nn
import CONFIG
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, df):
        self.sentiments = df['sentiment'].values
        self.reviews = df['review'].values
        self.tokenizer = CONFIG.Tokenizer
        self.max_len = CONFIG.max_len
    
    def __len__(self):
        return len(self.sentiments)
    
    def __getitem__(self, idx):
        sentiment = self.sentiments[idx]
        review = self.reviews[idx]
        input_ = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length= CONFIG.max_len,
            truncation=True
        )
        ids = input_['input_ids']
        mask = input_['attention_mask']
        token_type_ids = input_['token_type_ids']
        
        return  {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
            'sentiment' : torch.tensor(sentiment, dtype=torch.float)
        }

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        self.mask_idx = 0
        self.token_type_idx = 0
    
    def __call__(self, batch):
        ids = [item['ids'] for item in batch]
        mask = [item['mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        sentiment = [item['sentiment'].unsqueeze(0) for item in batch]
        sentiment = torch.cat(sentiment, dim=0)
        

        ids = torch.nn.utils.rnn.pad_sequence(
            ids,
            batch_first=True,
            padding_value=self.pad_idx
        )
        mask = torch.nn.utils.rnn.pad_sequence(
            mask,
            batch_first=True,
            padding_value=self.mask_idx
        )
        token_type_ids = torch.nn.utils.rnn.pad_sequence(
            token_type_ids,
            batch_first=True,
            padding_value=self.token_type_idx
        )
        

        return {
            'ids' : ids,
            'mask' : mask,
            'token_type_ids' : token_type_ids,
            'sentiment' : sentiment
        }
