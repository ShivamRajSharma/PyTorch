import spacy 
import CONFIG
import model
import torch 
import numpy as np

def predict(sentence):
    model = model.LSTMModel()
    model.load_state_dict(torch.load(CONFIG.Model_Path))

    word_to_idx = pickle.load(open('../input/word_to_idx.pickle'))
    pos_lb = pickle.load(open('../input/pos_lb.pickle'))
    ner_lb = pickle.load(open('../input/ner_lb.pickle'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sentence_idx = []
    tokenized_sentence = []
    tokenizer = spacy.load('en')
    for word in tokenizer(sentence):
        word = str(word.text().lower())
        tokenized_sentence.append(word)
        if word in word_to_idx:
            sentence_idx.append(word_to_idx[word])
        else:
            sentence_idx.append(word_to_idx['<UNK>'])
    
    sentence_idx = torch.tensor(sentence_idx, dtype=torch.long).unsqueeze(0)
    
    sentence_idx = sentence_idx.to(device)

    model = model.to(device)

    pos_tag , ner_tag = model(sentence_idx)

    pos_tag =  pos_tag.unsqueeze(0).argmax(1).item()
    ner_tag = ner_tag.unsqueeze(0).argmax(1).item()

    pos_out = pos_lb.inverse_transform(pos_tag)
    ner_out = ner_lb.inverse_transform(ner_tag)
    
    print(sentence)

    for num, word in enumerate(tokenized_sentence):
        print(f' WORD -> {word} | POS -> {pos_out[num]} | NER -> {ner_out[num]} \n' )

if __name__ == '__main__':
    sentence = str(input('ENTER A SENTENCE'))
    predict(sentence)

    