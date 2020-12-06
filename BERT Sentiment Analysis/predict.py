import config 
import BertSentimentModel
import torch
import torch.nn as nn

def predict(sentence):
    model = BertSentimentModel.BERTMODEL()

    model.load_state_dict(torch.load(config.Model_Path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    inputs = config.Tokenizer.encode_plus(
        sentence,
        None,
        add_special_tokens=True
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0)

    ids = ids.to(device)
    mask = mask.to(device)
    token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        output = model(ids, mask, token_type_ids)

    output = (torch.sigmoid(output[0]).item() > 0.5)*1

    if output == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negetive'

    print(f'REVIEW -> {sentence} | SENTIMENT -> {sentiment} \n')

if __name__ == "__main__":
    print('\n')
    sentence = str(input("Enter a movie review : "))
    print('\n')
    predict(sentence)

