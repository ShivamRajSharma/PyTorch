import CONFIG

import torch 
import spacy
from tqdm import tqdm
import torch.nn as nn 

class Vocabulary:
    def __init__(self):
        self.tokenizer = spacy.load('en_core_web_sm')
        self.word_to_idx = {"<UNK>": 0, "<SOS>": 1, "<EOS>": 2, "<PAD>": 3}
        self.idx_to_word = {0: "<UNK>", 1: "<SOS>", 2: "<EOS>", 3: "<PAD>"}
        self.word_feq = {}
        self.thresh =  CONFIG.num_word_threshold
    
    def vocab(self, datas):
        num = 4
        for data in datas:
            for sentence in tqdm(data, total=len(data)):
                for word in self.tokenizer(sentence):
                    word = str(word.text.lower())
                    if word not in self.word_to_idx:
                        if word not in self.word_feq:
                            self.word_feq[word] = 1
                            if self.word_feq[word] > self.thresh:
                                self.word_to_idx[word] = num
                                self.idx_to_word[num] = word
                                num += 1
                        else:
                            self.word_feq[word] += 1
                            if self.word_feq[word] > self.thresh:
                                self.word_to_idx[word] = num
                                self.idx_to_word[num] = word
                                num += 1
    
    def numericalize(self, sentence):
        sentence_idx = []
        for word in self.tokenizer(sentence):
            word = str(word.text.lower())
            if  word in self.word_to_idx:
                sentence_idx.append(self.word_to_idx[word])
            else:
                sentence_idx.append(self.word_to_idx["<UNK>"])
        return sentence_idx


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text = df.text.values
        self.headlines = df.headlines.values
        self.vocab = Vocabulary()
        self.vocab.vocab([self.text, self.headlines])
    
    def  __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        headline = self.headlines[idx]
        text_idx = self.vocab.numericalize(text)
        headline_idx = [self.vocab.word_to_idx["<SOS>"]]
        headline_idx += self.vocab.numericalize(headline)
        headline_idx.append(self.vocab.word_to_idx["<EOS>"])

        return {
            "text_idx" : torch.tensor(text_idx, dtype=torch.long),
            "headline_idx" : torch.tensor(headline_idx, dtype=torch.long)
        }

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        text = [item['text_idx'] for item in batch]
        headlines = [item['headline_idx'] for item in batch]
        padded_text = nn.utils.rnn.pad_sequence(
            text,
            batch_first=True,
            padding_value=self.pad_idx
        )
        padded_headlines = nn.utils.rnn.pad_sequence(
            headlines,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return {
            "text_idx" : padded_text,
            "headline_idx" : padded_headlines 
        }