import torch
import torch.nn as  nn

class Attention(nn.Module):
    def __init__(self, heads, input_dims):
        super(Attention, self).__init__()
        self.heads = heads
        self.input_dims = input_dims
        self.d = self.input_dims//self.heads
        self.key = nn.Linear(self.d, self.d)
        self.query = nn.Linear(self.d, self.d)
        self.value = nn.Linear(self.d, self.d)

        self.out = nn.Linear(self.d*self.heads, self.input_dims)
    
    def forward(self, query, key, value, mask):
        key_seq_len, query_seq_len, value_seq_len = key.shape[1], query.shape[1], value.shape[1]
        
        key = key.view(-1, key_seq_len, self.heads, self.d)
        query = query.view(-1, query_seq_len, self.heads, self.d)
        value = value.view(-1, value_seq_len, self.heads, self.d)

        key = self.key(key)
        query = self.query(query)
        value = self.value(value)

        score = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        if mask is not None:
            score.masked_fill(mask==0, float("-1e20"))
        score = torch.softmax(score/((self.d)**1/2), dim=-1)
        
        out = torch.einsum("bhqk,bkhd->bqhd", [score, value])
        out = out.reshape(-1, query_seq_len, self.heads*self.d)
        out = self.out(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, input_dims, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(heads, input_dims)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dims, input_dims*forward_expansion),
            nn.GELU(),
            nn.Linear(input_dims*forward_expansion, input_dims)
        )
        self.layer_norm1 = nn.LayerNorm(input_dims)
        self.layer_norm2 = nn.LayerNorm(input_dims)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        add =  self.dropout(self.layer_norm1(attention + query))
        fc = self.feed_forward(add)
        out =  self.dropout(self.layer_norm1(add + fc))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        max_len,
        heads,
        forward_expansion,
        num_layers,
        dropout
    ):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dims)
        self.positonal_emebedding = nn.Parameter(torch.zeros(1, max_len, embed_dims))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dims,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        seq_len = x.shape[1]
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positonal_emebedding[:, :seq_len, :]
        out = self.dropout(token_embedding + positional_embedding)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dims,
        heads,
        forward_expansion,
        dropout
    ):
        super(DecoderBlock, self).__init__()
        self.attention = Attention(heads, input_dims)
        self.transformer_block = TransformerBlock(
            input_dims,
            heads,
            dropout,
            forward_expansion
        )
        self.layer_norm = nn.LayerNorm(input_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        query, 
        key, 
        value, 
        src_mask, 
        causal_mask
    ):
        attention = self.attention(query, query, query, causal_mask)
        query = self.dropout(self.layer_norm(attention + query))
        out = self.transformer_block(query, key, value, src_mask)

        return out

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dims,
        max_len,
        heads,
        forward_expansion,
        num_layers,
        dropout
    ):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dims)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dims))
        self.layers = nn.Sequential(
            *[
                DecoderBlock(
                    embed_dims,
                    heads,
                    forward_expansion,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dims, vocab_size)
    
    def forward(self, x, encoder_output, src_mask, causal_mask):
        seq_len = x.shape[1]
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding[:, :seq_len, :]
        x = self.dropout(token_embedding + positional_embedding)
        for layer in self.layers:
            x = layer(
                x, 
                encoder_output, 
                encoder_output, 
                src_mask, 
                causal_mask
            )
        out = self.out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        out_vocab_size,
        max_len,
        embed_dims,
        pad_idx,
        heads,
        forward_expansion,
        num_layers,
        dropout, device
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            input_vocab_size,
            embed_dims,
            max_len,
            heads,
            forward_expansion,
            num_layers,
            dropout
        )

        self.decoder = Decoder(
            out_vocab_size,
            embed_dims,
            max_len,
            heads,
            forward_expansion,
            num_layers,
            dropout
        )
        self.device = device
        
        self.pad_idx = pad_idx
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    
    def pad_mask(self, inputs):
        mask = (inputs!=self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def causal_mask(self, targets):
        batch_size, seq_len = targets.shape
        mask =  torch.tril(torch.ones((batch_size, seq_len, seq_len))).unsqueeze(1)
        return mask
    
    def forward(self, inputs, target):
        pad_mask = self.pad_mask(inputs).to(self.device)
        causal_mask = self.causal_mask(target).to(self.device)
        encoder_output = self.encoder(inputs, pad_mask)
        decoder_out = self.decoder(target, encoder_output, pad_mask, causal_mask)
        return decoder_out


if __name__ == "__main__":
    #Depends on the Tokenizer
    input_vocab_size = 20
    out_vocab_size = 30

    #DEFAULT TRANSFORMERS PARAMETERS:-
    pad_idx = 0 
    embed_dims = 512
    num_layers = 1
    forward_expansion = 4
    heads = 8
    dropout = 0.1
    max_len = 512

    inputs = torch.randint(0, 20, (32, 200))
    targets = torch.randint(0, 30, (32, 100))

    model = Transformer(
        input_vocab_size,
        out_vocab_size,
        max_len,
        embed_dims,
        pad_idx,
        heads,
        forward_expansion,
        num_layers,
        dropout
    )
    y = model(inputs, targets)
    print(y.shape)





