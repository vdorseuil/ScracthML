import numpy as np
import torch
import torch.nn as nn


import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.compute()

    def SinPos(self, i: int, pos: int):
        if i % 2 == 0:
            return np.sin(pos / 10000 ** (2 * i / self.d_model))
        else:
            return np.cos(pos / 10000 ** (2 * i / self.d_model))

    def compute(self):
        self.Mat = torch.Tensor([[self.SinPos(i, pos) for i in range(self.d_model)] for pos in range(self.max_length)])


    def forward(self, x):
        return self.Mat[:x.shape[-1], :]


class Embedding(nn.Module):
    def __init__(self, d_model, max_length, n_embedding, dropout):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model
        self.n_embedding = n_embedding
        self.embedding = nn.Embedding(num_embeddings=n_embedding, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_length=max_length)
        self.dropout = nn.Dropout(p=dropout)
        pass

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding(x)
        return self.dropout(x)


class SingleHeadAttention(nn.Module):
    def __init__(self, dk: int, dv: int, d_model: int):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.d_model = d_model
        self.K = nn.Linear(in_features=d_model, out_features=dk)
        self.Q = nn.Linear(in_features=d_model, out_features=dk)
        self.V = nn.Linear(in_features=d_model, out_features=dv)

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor = None, mask=None):
        Kx = self.K(x_encoder) if x_encoder is not None else self.K(x)
        Vx = self.V(x_encoder) if x_encoder is not None else self.V(x)
        Qx = self.Q(x)
        QK = torch.matmul(Qx, Kx.transpose(-2, -1)) / np.sqrt(self.dk)
        if mask is not None:
            QK = QK + mask
        QK = torch.softmax(QK, dim=-1)
        return torch.matmul(QK, Vx)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dk: int, dv: int, d_model: int):
        super().__init__()
        assert num_heads * dv == d_model, "num_heads * dv should be equal to the model dim"
        self.attention_heads = nn.ModuleList([SingleHeadAttention(dk=dk, dv=dv, d_model=d_model) for _ in range(num_heads)])
        self.WO = nn.Linear(in_features=num_heads * dv, out_features=d_model)

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor = None, mask=None):
        outputs = [head(x, x_encoder, mask) for head in self.attention_heads]
        x = torch.cat(outputs, dim=-1)
        x = self.WO(x)
        return x
    

class SublayerConnection(nn.Module): #Dropout, Add and Norm
    def __init__(self, features, dropout, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.dropout = nn.Dropout(p=dropout)

        self.eps = eps

    def forward(self, x, sublayer_output):
        x = x + self.dropout(sublayer_output) #dropout and add
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2 #norm
        return x



class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, d_model, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, dk=dk, dv=dv, d_model=d_model)
        self.sublayer1 = SublayerConnection(features=d_model, dropout=dropout)
        self.sublayer2 = SublayerConnection(features=d_model, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x, mask=None):
        attention = self.attention(x=x, mask=mask)
        x = self.sublayer1(x, attention)
        feedforward = self.ff(x)
        x = self.sublayer2(x, feedforward)
        return x


class Encoder(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, d_model, dropout, num_encoders):
        super().__init__()
        self.encoders_list = [EncoderBlock(num_heads=num_heads, dk=dk, dv=dv, d_ff=d_ff, d_model=d_model, dropout=dropout) for _ in range(num_encoders)]
        self.encoders = nn.ModuleList(self.encoders_list)

    def forward(self, x, mask=None):
        for encoder in self.encoders_list:
            x = encoder(x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, d_model, dropout):
        super().__init__()
        self.masked_attention = MultiHeadAttention(num_heads=num_heads, dk=dk, dv=dv, d_model=d_model)
        self.mixed_attention = MultiHeadAttention(num_heads=num_heads, dk=dk, dv=dv, d_model=d_model)
        self.sublayer1 = SublayerConnection(features=d_model, dropout=dropout)
        self.sublayer2 = SublayerConnection(features=d_model, dropout=dropout)
        self.sublayer3 = SublayerConnection(features=d_model, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x, x_encoder, causal_mask=None, mixed_mask=None):
        attention = self.masked_attention(x, mask=causal_mask)
        x = self.sublayer1(x, attention)
        attention = self.mixed_attention(x, x_encoder=x_encoder, mask=mixed_mask)
        x = self.sublayer2(x, attention)
        feedforward = self.ff(x)
        x = self.sublayer3(x, feedforward)
        return x

class Decoder(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, d_model, dropout, num_decoders):
        super().__init__()
        decoders_list = [DecoderBlock(num_heads=num_heads, dk=dk, dv=dv, d_ff=d_ff, d_model=d_model, dropout=dropout) for _ in range(num_decoders)]
        self.decoders = nn.ModuleList(decoders_list)

    def forward(self, x, x_encoder, causal_mask=None, mixed_mask=None):
        for decoder in self.decoders:
            x = decoder(x, x_encoder, causal_mask, mixed_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, max_length, vocab_size, num_out, num_heads, dv, dk, d_ff, dropout, num_encoders, num_decoders):
        super().__init__()
        self.embedding = Embedding(d_model, max_length, n_embedding=vocab_size, dropout=dropout)
        self.encoder = Encoder(num_heads, dk, dv, d_ff, d_model, dropout, num_encoders)
        self.decoder = Decoder(num_heads, dk, dv, d_ff, d_model, dropout, num_decoders)
        self.linear = nn.Linear(in_features=d_model, out_features=num_out)
        self.ff_mask = torch.zeros(max_length, max_length) + torch.triu(torch.full((max_length, max_length), float("-inf")), diagonal=1)

    def forward(self, input, output):
        input_embed = self.embedding(input)
        output_embed = self.embedding(output)
        x_encoder = self.encoder(input_embed)
        x = self.decoder(output_embed, x_encoder, causal_mask=self.ff_mask)
        x = self.linear(x)
        return x
    

    def generate(self, input, max_gen_length, start_token, end_token): #greed decoding
        self.eval()
        input_embed = self.embedding(input)
        x_encoder = self.encoder(input_embed)

        generated_tokens = [start_token]
        generated_tokens_probas = [1]

        for _ in range(max_gen_length):
            output = torch.tensor(generated_tokens).unsqueeze(0) # size [1, sequence_length]
            out_embed = self.embedding(output)
            causal_mask = self.ff_mask[:out_embed.size(1), :out_embed.size(1)]
            x = self.decoder(out_embed, x_encoder, causal_mask=causal_mask)
            x = self.linear(x)
            probas = torch.softmax(x, dim=-1)
            max_proba, next_token = torch.max(probas[:, -1, :], dim=-1) #greedy decoding : only max_proba
            generated_tokens.append(next_token.item())
            generated_tokens_probas.append(max_proba.item())
            if next_token == end_token:
                break
            
        return generated_tokens, generated_tokens_probas