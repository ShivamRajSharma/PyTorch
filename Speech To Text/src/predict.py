import torch 
import pickle 
import torch.nn as nn
import MODEL
import CONFIG
import torchaudio

def predict(path):
    model = MODEL.ASRModel(
        input_channel=1, 
        out_channel=CONFIG.out_channel, 
        kernel_size=CONFIG.kernel_size, 
        padding=CONFIG.padding,
        num_inception_block=CONFIG.num_inception_block,
        squeeze_dims=CONFIG.squeeze_dims,
        rnn_input_dims=CONFIG.rnn_input_dims,
        hidden_dims=CONFIG.hidden_dims,
        num_layers=CONFIG.num_layers,
        bidirectional=CONFIG.bidirectional,
        dropout=CONFIG.dropout,
        num_classes=CONFIG.num_classes
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(CONFIG.MODEL_PATH))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        char_to_idx = pickle.load(open('../input/char_to_idx.pickle', 'rb'))
        idx_to_char = pickle.load(open('../input/idx_to_char.pickle', 'rb'))
        waveform, sample_rate = torchaudio.load(path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=CONFIG.n_filters
        )(waveform).squeeze(0)

        mel_spectrogram = mel_spectrogram.to(device).unsqueeze(0).unsqueeze(1)

        output = model(mel_spectrogram)
        output = nn.functional.log_softmax(output, dim=2)
        output = torch.argmax(output, dim=2).squeeze(0)
        output = output.detach().cpu().numpy()
        text = []
        for num, idx in enumerate(output):
            if idx != CONFIG.blank_idx and idx != output[num-1]:
                word = idx_to_char[idx]
                if word == '<SPACE>':
                    text.append(' ')
                else:
                    text.append(word)

    text = ''.join(text)

    print(f'AUDIO FILE SAYS -> {text}')

if __name__ == "__main__":
    path = '../input/LibriSpeech/train-clean-100/19/198/19-198-0001.flac'
    predict(path)
