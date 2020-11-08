import torch 
import torch.nn as nn 
import sys

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class NER(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        hidden_dims,
        num_layers,
        num_classes,
        dropout,
        bidirectional,
        tag_to_idx,
        device
    ):
        super(NER, self).__init__()
        self.tag_to_idx = tag_to_idx
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embed_dims)
        self.rnn = nn.LSTM(
            embed_dims,
            hidden_dims,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.norm_layer1 = nn.LayerNorm(embed_dims)
        self.rnn_out_dims = hidden_dims*2 if bidirectional else hidden_dims
        self.norm_layer2 = nn.LayerNorm(self.rnn_out_dims)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_out_dims, int(hidden_dims/4)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dims/4), num_classes)
        )

        self.transition = nn.Parameter(
            torch.randn(self.num_classes, self.num_classes)
        )
        self.device = device
        self.transition.data[tag_to_idx['start_tag'], :] = -10000
        self.transition.data[:, tag_to_idx['stop_tag']] = -10000

    
    def forward_algo(self, feats):
        init_alphas = torch.full((1, self.num_classes), -10000.).to(self.device)
        init_alphas[0][self.tag_to_idx['start_tag']] = 0
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.num_classes):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.num_classes)
                trans_score = self.transition[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transition[self.tag_to_idx['stop_tag']]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def sentence_score(self, feats, tags):
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_idx['start_tag']], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score += self.transition[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transition[self.tag_to_idx['stop_tag'], tags[-1]]
        return score

    
    def veterbi(self, feats):
        backpointer = []

        init_vvars = torch.full((1, self.num_classes), -10000.).to(self.device)
        init_vvars[0][self.tag_to_idx['start_tag']] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            veterbivars_t = []
            
            for next_tag in range(self.num_classes):
                next_tag_var = forward_var + self.transition[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                veterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(veterbivars_t) + feat).view(1, -1)
            backpointer.append(bptrs_t)
        

        terminal_var = forward_var + self.transition[self.tag_to_idx['stop_tag']]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointer):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        best_path.reverse()
        return path_score, best_path



    def base_model(self, x):
        x = self.dropout(self.norm_layer1(self.embedding(x))).view(x.shape[-1], 1, -1)
        x, (hidden, cell) = self.rnn(x)
        x = x.view(x.shape[0], x.shape[-1])
        x = self.dropout(self.norm_layer2(x))
        x = self.classifier(x)
        return x
    
    def training_fn(self, x, tags):
        feats = self.base_model(x)
        forward_score = self.forward_algo(feats)
        gold_score = self.sentence_score(feats, tags)
        return forward_score - gold_score
    
    def forward(self, x):
        feats = self.base_model(x)
        score, tag_seq = self.veterbi(feats)
        return score, tag_seq