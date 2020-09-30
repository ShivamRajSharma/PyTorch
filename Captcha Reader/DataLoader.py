import CONFIG
from PIL import Image
import  os
import torch 
import torch.nn as nn 
import numpy as np

class Vocabulary:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def vocab(self, data):
        num = 0
        for image_title in data:
            image_title = str(image_title).lower()
            for letter in image_title:
                if letter not in self.word_to_idx:
                    self.word_to_idx[letter] = num
                    self.idx_to_word[num] = letter
                    num += 1
    
    def numericalize(self, image_title):
        image_title_idx = []
        for letter in str(image_title).lower():
            if letter in self.word_to_idx:
                image_title_idx.append(self.word_to_idx[letter])
        return image_title_idx


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, transforms):
        self.image_names = os.listdir(CONFIG.INPUT_PATH)
        self.vocab = Vocabulary()
        self.vocab.vocab(self.image_names)
        self.transforms =  transforms

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        target = self.vocab.numericalize(image_name.split('.')[0])
        image = np.array(Image.open(os.path.join(CONFIG.INPUT_PATH, image_name)).convert('RGB'))
        image = self.transforms(image=image)['image']
        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)
        target_len = len(target)
        out_len =  image.shape[2] // 5

        return {
            'image' : image,
            'target' : torch.tensor(target, dtype=torch.float),
            'out_len' : out_len,
            'target_len' : target_len
        }
