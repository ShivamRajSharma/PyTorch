import spacy
import torch 
import torch.nn as nn
from tqdm import tqdm

class Vocabulary:
    def __init__(self):
        self.word_to_idx = {'<UNK>' : 0, '<PAD>' : 1}
        self.idx_to_word = {0 : '<UNK>', 1 : '<PAD>' }
        self.tokenizer = spacy.load('en_core_web_sm')
    
    def vocab_gen(self, data):
        num = 2
        for sentence in tqdm(data, total=len(data)):
            for word in self.tokenizer(sentence):
                word = str(word.text.lower)
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = num
                    self.idx_to_word[num] = word
                    num += 1
    
    def numericalize(self, sentence):
        sentence_idx = []
        for word in self.tokenizer(sentence):
            word = str(word.text.lower)
            if word in self.word_to_idx:
                sentence_idx.append(self.word_to_idx[word])
            else:
                sentence_idx.append(self.word_to_idx['<UNK>'])
        return sentence_idx


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        self.que_1 = df.question1.values
        self.que_2 = df.question2.values
        self.labels =  df.is_duplicate.values
        self.vocab =  Vocabulary()
        self.vocab.vocab_gen(self.que_1)
        self.vocab.vocab_gen(self.que_2)
    
    def __len__(self):
        return len(self.que_1)
    
    def __getitem__(self, idx):
        que_1 = self.que_1[idx]
        que_2 = self.que_2[idx]
        label = self.labels[idx]

        que_1_idx = self.vocab.numericalize(que_1)
        que_2_idx = self.vocab.numericalize(que_2)

        return {
            'que_1' : que_1_idx,
            'que_2' : que_2_idx,
            'target' : label
        }


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        max_len = -10
        for item in batch:
            len_que_pair = max(len(item['que_1']), len(item['que_2']))
            if len_que_pair > max_len:
                max_len = len_que_pair

        que_1_batch = []
        que_2_batch = []
        labels_batch = []

        for item in batch:
            que_1 = item['que_1']
            que_2 = item['que_2']
            label = item['target']
            que_1_pad_len = max_len - len(que_1)
            que_2_pad_len = max_len - len(que_2)
            que_1 += [self.pad_idx]*que_1_pad_len
            que_2 += [self.pad_idx]*que_2_pad_len
            que_1_batch.append(que_1)
            que_2_batch.append(que_2)
            labels_batch.append(label)
        return {
            'que_1' : torch.tensor(que_1_batch, dtype=torch.long),
            'que_2' : torch.tensor(que_2_batch, dtype=torch.long),
            'target' : torch.tensor(labels_batch, dtype=torch.float)
        }

if __name__ == "__main__":
    pass
