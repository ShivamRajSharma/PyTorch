import CONFIG 
import model_dispatcher
import torch
import torch.nn as nn

def predict(sentence):
    model = model_dispatcher.BERTMODEL()

    model.load_state_dict(torch.load(CONFIG.Model_Path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    inputs = CONFIG.Tokenizer.encode_plus(
        sentence,
        None,
        add_special_tokens=True
    )

    ids = input['input_ids'].unsqueeze(0)
    mask = input['attention_mask'].unsqueeze(0)
    token_type_ids = input['token_type_ids'].unsqueeze(0)

    ids = ids.to(device)
    mask = mask.to(device)
    token_type_ids = token_type_ids.to(device)

    output = model(ids, mask, token_type_ids)

    output = (torch.sigmoid(output).item() > 0.5)*1

    if output == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negetive'

    print(f'REVIEW -> {sentence} | SENTIMENT -> {sentiment}')




