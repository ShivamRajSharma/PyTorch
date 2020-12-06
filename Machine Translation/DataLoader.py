import gc
import os 
import pickle
import torch 
import torch.nn as nn 
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import spacy

class Vocabulary:
    def __init__(self, tokenizer, language):
        self.idx_to_word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>' :3}
        self.tokenizer = tokenizer
        self.language = language
    
    def build_vocab(self, sentences):
        idx = 4
        if os.path.exists(f'input/{self.language}_word_to_idx.pickle') and os.path.exists(f'input/{self.language}_idx_to_word.pickle'):
            print(f'------ [INFO] LOADING PRECOMPUTED {self.language} TOKENIZER --------')
            self.word_to_idx = pickle.load(open('input/' + self.language + '_word_to_idx.pickle', 'rb'))
            self.idx_to_word = pickle.load(open('input/' +  self.language +'_idx_to_word.pickle', 'rb'))
        else:
            for sentence in tqdm(sentences, total=len(sentences)):
                for word in self.tokenizer(sentence):
                    word = str(word.text.lower())
                    if self.word_to_idx.get(word) is None:
                        self.word_to_idx[word] = idx
                        self.idx_to_word[idx] = word
                        idx += 1
    
    def token(self, sentence):
        x = [word.text.lower() for word in self.tokenizer(sentence)]
        return x
    
    def numericalize(self, text):
        tokenized_text = self.token(text)
        return [
            self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<UNK>'] for word in tokenized_text
        ]
            

class TranslationDataset(Dataset):
    def __init__(self, english, german):
        self.english = english
        self.german = german
        self.german_preprocessing = Vocabulary(spacy.load('de_core_news_sm'), 'german') 
        self.english_preprocessing = Vocabulary(spacy.load('en_core_web_sm'), 'english')
        print('------[INFO] TOKENIZING GERMAN SENTENCES -------\n')
        self.german_preprocessing.build_vocab(self.german)
        print('\n------[INFO] TOKENIZING ENGLISH SENTENCES ------')
        self.english_preprocessing.build_vocab(self.english)

    
    def __len__(self):
        return len(self.english)
    
    def __getitem__(self, idx):
        german = self.german[idx]
        english = self.english[idx]
        german_to_idx = [self.german_preprocessing.word_to_idx['<SOS>']]
        german_to_idx += self.german_preprocessing.numericalize(german)
        german_to_idx.append(self.german_preprocessing.word_to_idx['<EOS>'])
        english_to_idx = [self.english_preprocessing.word_to_idx['<SOS>']]
        english_to_idx += self.english_preprocessing.numericalize(english)
        english_to_idx.append(self.english_preprocessing.word_to_idx['<EOS>'])
        
        return {
            'german_idx' : torch.tensor(german_to_idx, dtype=torch.long), 
            'english_idx' : torch.tensor(english_to_idx, dtype=torch.long)
        }

class MyCollate:
    def __init__(self, eng_pad_idx, grm_pad_idx):
        self.eng_pad_idx = eng_pad_idx
        self.grm_pad_idx = grm_pad_idx
    
    def __call__(self, batch):
        german_idx = [item['german_idx'] for item in batch]
        german_idx = pad_sequence(
            german_idx, 
            batch_first=True, 
            padding_value=self.grm_pad_idx
        )

        english_idx = [item['english_idx'] for item in batch]
        english_idx = pad_sequence(
            english_idx,
            batch_first=True,
            padding_value=self.eng_pad_idx
        )

        return {
            'german_idx' : german_idx, 
            'english_idx' : english_idx
        }

if __name__ == "__main__":
    english_path = 'input/train.en'
    german_path = 'input/train.de'
    dataset = TranslationDataset(english_path, german_path)
    pad_idx = dataset.english_preprocessing.word_to_idx['<PAD>']
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=4,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    for idx, d in enumerate(loader):
        print(d['english_idx'])
        print(d['german_idx'])
