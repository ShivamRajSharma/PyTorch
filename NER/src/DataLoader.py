import torch 
import spacy
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self):
        self.word_to_idx = {'<PAD>' : 0, '<UNK>' : 1}
        self.tokenizer = spacy.load('en')
    
    def vocab(self, sentences):
        num = 2
        for sentence in sentences:
            for word in self.tokenizer(sentence):
                word = str(word.text.lower())
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = num
                    num += 1
    
    def numercalize(self, sentence):
        sentence_idx = []
        for word in self.tokenizer(sentence):
            word =  str(word.test.lower())
            if word in self.word_to_idx:
                sentence_idx.append(self.word_to_idx[word])
            else:
                sentence_idx.append(self.word_to_idx['<UNK>'])
        return sentence_idx
            

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        self.sentences = df.sentence.values
        self.pos = df.pos.values
        self.ner = df.ner.values
        self.vocab = Vocab()
        self.vocab.vocab(self.sentences)
    
    def __len__(self):
        return len(self.sentence)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        pos_tag = self.pos[idx]
        ner_tag = self.ner[idx]
        sentence_idx = self.vocab.numercalize(sentence)

        return {
            'sentence' : sentence,
            'sentence_idx' : torch.tensor(sentence_idx, dtype=torch.long),
            'pos_tag' : torch.tensor(pos_tag, dtype=torch.float),
            'ner_tag' : torch.tensor(ner_tag, dtype=torch.float)
        }


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        sentence = [item['sentence'].unsqueeze(0) for item in batch]
        sentence_idx = [item['sentence_idx'] for item in batch]
        pos_tag = [item['pos_tag'] for item in batch]
        ner_tag = [item[ner_tag] for item in batch]

        sentence_idx = pad_sequence(
            sentence_idx,
            batch_first=True,
            padding_value=self.pad_idx
        )

        pos_tag = pad_sequence(
            pos_tag,
            batch_first=True,
            padding_value=self.pad_idx
        )

        ner_tag = pad_sequence(
            ner_tag,
            batch_first=True,
            padding_value=self.pad_idx
        )

        sentence = torch.cat(sentence, dim=0)

        return {
            'sentence' : sentence,
            'sentence_idx' : torch.tensor(sentence_idx, dtype=torch.long),
            'pos_tag' : torch.tensor(pos_tag, dtype=torch.float),
            'ner_tag' : torch.tensor(ner_tag, dtype=torch.float)
        }