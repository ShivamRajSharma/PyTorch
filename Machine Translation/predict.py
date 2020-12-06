import warnings
warnings.filterwarnings('ignore')

import TranslationModel 
import CONFIG

import pickle
import spacy
import torch
import torch.nn as nn 

def predict(sentence):
    max_len = 50
    english_word_to_idx = pickle.load(open('input/english_word_to_idx.pickle', 'rb'))
    german_word_to_idx = pickle.load(open('input/german_word_to_idx.pickle', 'rb'))

    english_idx_to_word = pickle.load(open('input/english_idx_to_word.pickle', 'rb'))
    german_idx_to_word = pickle.load(open('input/german_idx_to_word.pickle', 'rb'))
    
    tokenizer = spacy.load('de_core_news_sm')


    encoder = TranslationModel.Encoder(
        vocab_size=len(german_word_to_idx),
        embedding_size=CONFIG.encoder_embed_dims,
        hidden_size=CONFIG.encoder_hidden_dims,
        num_hidden_layer=CONFIG.encoder_num_layers,
        dropout_ratio=CONFIG.encoder_dropout
    )
    

    decoder = TranslationModel.Decoder(
        vocab_size=len(english_word_to_idx),
        embedding_dims=CONFIG.decoder_embed_dims,
        hidden_size=CONFIG.decoder_hidden_dims,
        num_hidden_layer=CONFIG.decoder_num_layers,
        dropout_ratio=CONFIG.decoder_dropout,
        output_size=len(english_word_to_idx),
    )


    model = TranslationModel.Encoder_Decoder(encoder, decoder, len(english_word_to_idx))
    model.load_state_dict(torch.load('model/model.bin'))


    tokenized_sentence = ['<SOS>']
    german_idx = [german_word_to_idx['<SOS>']]
    for word in tokenizer(sentence):
        word = str(word.text.lower())
        tokenized_sentence.append(word)
        if word in german_word_to_idx:
            german_idx.append(german_word_to_idx[word])
        else:
            german_idx.append(german_word_to_idx['<UNK>'])
    tokenized_sentence.append('<EOS>')
    german_idx.append(german_word_to_idx['<EOS>'])
    
    german_idx = torch.tensor(german_idx, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(german_idx)
        predicted_sentence = ['<SOS>']

        for _ in range(max_len):
            previous_word = torch.tensor([english_word_to_idx[predicted_sentence[-1]]])
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = torch.softmax(output, dim=-1)
            best_guess = best_guess.argmax(-1).item()
            if best_guess == english_word_to_idx['<EOS>']:
                break
            predicted_sentence.append(english_idx_to_word[best_guess])
            
    predicted_sentence = ' '.join(predicted_sentence[1:])

    return predicted_sentence

if __name__ == '__main__':
    sentence = str(input("Write Something in German :"))
    translation = predict(sentence)
    print(f'English Translation : {translation}')
