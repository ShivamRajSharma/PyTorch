import torch 
import CONFIG

import numpy as np
import torch.nn as nn
import torchaudio

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data, word_to_idx_map, transforms=None):
        self.path = data[:, 0]
        self.target_text = data[:, 1]
        self.word_to_idx_map = word_to_idx_map
        self.transforms = transforms
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        path = self.path[idx]
        target_text = self.target_text[idx]
        target = []

        for alphabet in target_text:
            if alphabet == ' ':
                target.append(self.word_to_idx_map['<SPACE>'])
            else:
                target.append(self.word_to_idx_map[alphabet])
        
        waveform, sample_rate = torchaudio.load(path)

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=CONFIG.n_filters
        )(waveform).squeeze(0)

        if self.transforms:
            for transform in self.transforms:
                a = np.random.randint(0, 100)
                if a < CONFIG.transform_threshold*100 :
                    mel_spectrogram = transform(mel_spectrogram).squeeze(0)

        target_len  = len(target)
        mel_len = mel_spectrogram.shape[-1]//3

        return {
            'mel_spect' : mel_spectrogram,
            'mel_len' : mel_len,
            'target' : torch.tensor(target, dtype=torch.float),
            'target_len' : target_len
        }
        

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        mel_spect = [item['mel_spect'] for item in batch]
        max_len = max([item['mel_spect'].shape[-1] for item in batch])
        target = [item['target'] for item in batch]
        mel_len = [item['mel_len'] for item in batch]
        target_len = [item['target_len'] for item in batch]

        mel_spect_ = torch.zeros((len(batch), 1, CONFIG.n_filters, max_len))
        for num, mel in enumerate(mel_spect):
            mel_spect_[num, 0, :, :mel.shape[-1]] = mel
        
        target = torch.nn.utils.rnn.pad_sequence(
            target,
            batch_first=True,
            padding_value=28
        )
        return {
            'mel_spect' : mel_spect_,
            'target' : target,
            'mel_len' : torch.tensor(mel_len),
            'target_len' : torch.tensor(target_len)
        }



if __name__ == "__main__":
    import os 
    import glob
    import pickle
    import numpy as np

    path = 'input/LibriSpeech/train-clean-100/19/198'
    txt = glob.glob(path + '/*txt')[0]
    f = open(txt).read().strip().split('\n')
    path_and_label = []
    for line in f:
        audio_name = line.split()[0]
        audio_file_full_path = os.path.join(path, audio_name + '.flac')
        label = ' '.join(line.split()[1:])
        label = label.lower()
        path_and_label.append([audio_file_full_path, label])
    path_and_label = np.array(path_and_label)
    
    char_to_idx = pickle.load(open('input/char_to_idx.pickle', 'rb'))
    data_loader = DataLoader(path_and_label, char_to_idx, None)
    data_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=8,
        collate_fn=MyCollate(28)
    )

    for data in data_loader:
        mel_spect = data['mel_spect']
        target = data['target']
        mel_len = data['mel_len']
        target_len = data['target_len']





            
