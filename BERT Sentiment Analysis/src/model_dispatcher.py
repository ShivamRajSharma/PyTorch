import torch
import torch.nn as nn 
import CONFIG
import transformers

class BERTMODEL(nn.Module):
    def __init__(self):
        super(BERTMODEL, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(CONFIG.BERT_Path)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        _, x = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        
        out = self.fc(self.dropout(x))
        
        return out

if __name__ == '__main__':
    a = torch.randint(0, 2, (2, 100))
    model = BERTMODEL()
    y = model(a, a, a)