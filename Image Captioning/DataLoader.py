import torch 
import torch.nn as nn
import spacy
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class Vocabulary:
    def __init__(self):
        self.word_to_idx = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.idx_to_word = {0:'<PAD>', 1:'<SOS>', 2:'<EOS>', 3:'<UNK>'}
        self.tokenizer = spacy.load('en_core_web_sm')
    
    def vocab(self, all_captions):
        idx = 4
        for caption in tqdm(all_captions, total=len(all_captions)):
            for word in self.tokenizer(caption):
                word = str(word.text.lower())
                if self.word_to_idx.get(word) is None:
                    self.word_to_idx[word] = idx 
                    self.idx_to_word[idx] = word
                    idx += 1
    
    def tokenize_sentence(self, sentence):
        return [str(word.text.lower()) for word in self.tokenizer(sentence)]

    def numericalize(self, sentence):
        tokenized_sentence = self.tokenize_sentence(sentence)
        return [self.word_to_idx[word] if word in self.word_to_idx else self.word_to_idx['<UNK>'] for word in tokenized_sentence]


        
class DataLoader(Dataset):
    def __init__(self, df, transform):
        super(DataLoader, self).__init__()
        self.captions = df['caption'].values
        self.image_name = df['image'].values
        self.vocab = Vocabulary()
        self.path = 'input/images'
        self.vocab.vocab(self.captions)
        self.transform = transform

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_path = self.image_name[idx]
        image = np.array(Image.open(os.path.join(self.path, image_path)).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)['image']

        idx_caption = [self.vocab.word_to_idx['<SOS>']]
        idx_caption += self.vocab.numericalize(caption)
        idx_caption.append(self.vocab.word_to_idx['<EOS>'])

        return {
            'Images' : image,
            'Captions' : torch.tensor(idx_caption, dtype=torch.long)
        }

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        images = [item['Images'].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item['Captions'] for item in batch]
        padded_caption = pad_sequence(
            captions,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return {
            'Images' : images,
            'Captions' : padded_caption
        }

if __name__ == "__main__":
    pass
