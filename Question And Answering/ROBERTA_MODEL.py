import CONFIG

import torch 
import torch.nn as nn
import transformers

class QnAModel(nn.Module):
    def __init__(self, dropout):
        super(QnAModel, self).__init__()
        self.base_model = transformers.RobertaModel.from_pretrained(CONFIG.ROBERTA_PATH)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        base_out, _ = self.base_model(ids, mask, token_type_ids)
        base_out = self.dropout(base_out)
        fc = self.fc(base_out)
        start, end = fc.split(split_size=1, dim=-1)
        start = start.squeeze(-1)
        end = end.squeeze(-1)
        
        return start, end