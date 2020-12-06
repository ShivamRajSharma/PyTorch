import os
import torch 
import spacy
import pickle
from tqdm import tqdm
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self):
        self.word_to_idx = {'<PAD>' : 0, '<UNK>' : 1}
    
    def vocab(self, sentences):
        num = 2
        if os.path.exists('input/word_to_idx.pickle'):
            print('--- [INFO] LOADING PRECOMPUTED TOKENIZER ---')
            self.word_to_idx = pickle.load(open('input/word_to_idx.pickle', 'rb'))
        else:
            for sentence in tqdm(sentences, total=len(sentences)):
                for word in sentence:
                    word = word.lower()
                    if word not in self.word_to_idx:
                        self.word_to_idx[word] = num
                        num += 1

    def numercalize(self, sentence):
        sentence_idx = []
        for word in sentence:
            word =  word.lower()
            if word in self.word_to_idx:
                sentence_idx.append(self.word_to_idx[word])
            else:
                sentence_idx.append(self.word_to_idx['<UNK>'])
        return sentence_idx
            

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentence, pos):
        self.sentences = sentence
        self.pos = pos
        self.vocab = Vocabulary()
        self.vocab.vocab(self.sentences)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        pos = self.pos[idx]
        sentence_idx = self.vocab.numercalize(sentence)

        return {
            'sentence' : sentence,
            'sentence_idx' : torch.tensor(sentence_idx, dtype=torch.long),
            'pos' : torch.tensor(pos, dtype=torch.long),
        }


class MyCollate:
    def __init__(self, pad_idx, pos_pad_idx):
        self.pad_idx = pad_idx
        self.pos_pad_idx = pos_pad_idx

    def __call__(self, batch):
        sentence = [item['sentence'] for item in batch]
        sentence_idx = [item['sentence_idx'] for item in batch]
        pos = [item['pos'] for item in batch]

        sentence_idx = pad_sequence(
            sentence_idx,
            batch_first=True,
            padding_value=self.pad_idx
        )

        pos = pad_sequence(
            pos,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return {
            'sentence' : sentence,
            'sentence_idx' : sentence_idx,
            'pos' : pos,
        }
