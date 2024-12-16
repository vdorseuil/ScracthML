import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, batch_size, model_dim, max_length):
        super().__init__()
        self.model_dim = model_dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.compute()

    def SinPos(self, i: int, pos: int):
        if i % 2 == 0:
            return np.sin(pos / 10000 ** (2 * i / self.model_dim))
        else:
            return np.cos(pos / 10000 ** (2 * i / self.model_dim))

    def compute(self):
        Mat = torch.Tensor([[self.SinPos(i, pos) for i in range(self.model_dim)] for pos in range(self.max_length)])
        self.Mat = Mat.expand(self.batch_size, -1, -1)

    def forward(self, x):
        with torch.no_grad():
            return self.Mat


class Embedding(nn.Module):
    def __init__(self, batch_size, model_dim, max_length, n_embedding):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_dim = model_dim
        self.n_embedding = n_embedding
        self.embedding = torch.nn.Embedding(num_embeddings=n_embedding, embedding_dim=model_dim)
        self.pos_encoding = PositionalEncoding(batch_size=batch_size, model_dim=model_dim, max_length=max_length)
        pass

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding(x)
        return x


class SingleHeadAttention(nn.Module):
    def __init__(self, dk: int, dv: int, model_dim: int, mask: torch.Tensor = None):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.model_dim = model_dim
        self.K = nn.Linear(in_features=model_dim, out_features=dk)
        self.Q = nn.Linear(in_features=model_dim, out_features=dk)
        self.V = nn.Linear(in_features=model_dim, out_features=dv)
        self.mask = mask

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor = None):
        Kx = self.K(x_encoder) if x_encoder is not None else self.K(x)
        Vx = self.V(x_encoder) if x_encoder is not None else self.V(x)
        Qx = self.Q(x)
        QK = torch.matmul(Qx, Kx.transpose(-2, -1)) / np.sqrt(self.dk)
        if self.mask is not None:
            QK += self.mask
        QK = torch.softmax(QK, dim=-1)
        return torch.matmul(QK, Vx)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dk: int, dv: int, model_dim: int, mask=None):
        super().__init__()
        assert num_heads * dv == model_dim, "num_heads * dv should be equal to the model dim"
        self.attention_heads = nn.ModuleList([SingleHeadAttention(dk=dk, dv=dv, model_dim=model_dim, mask=mask) for _ in range(num_heads)])
        self.WO = nn.Linear(in_features=num_heads * dv, out_features=model_dim)
        self.mask = mask

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor = None):
        outputs = [head(x, x_encoder) for head in self.attention_heads]
        x = torch.cat(outputs, dim=-1)
        x = self.WO(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, model_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, dk=dk, dv=dv, model_dim=model_dim)
        self.layerNorm1 = nn.LayerNorm(normalized_shape=model_dim)
        self.layerNorm2 = nn.LayerNorm(normalized_shape=model_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=model_dim),
        )

    def forward(self, x):
        attention = self.attention(x)
        x = self.layerNorm1(x + attention)
        feedforward = self.ff(x)
        x = self.layerNorm2(x + feedforward)
        return x


class Encoder(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, model_dim, dropout, num_encoders):
        super().__init__()
        self.encoders_list = [
            EncoderBlock(num_heads=num_heads, dk=dk, dv=dv, d_ff=d_ff, model_dim=model_dim, dropout=dropout) for _ in range(num_encoders)
        ]
        self.encoders = nn.Sequential(*self.encoders_list)

    def forward(self, x):
        x = self.encoders(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, model_dim, dropout, max_length):
        super().__init__()
        self.mask = torch.zeros(max_length, max_length) + torch.triu(torch.full((max_length, max_length), float("-inf")), diagonal=1)
        self.masked_attention = MultiHeadAttention(num_heads=num_heads, dk=dk, dv=dv, model_dim=model_dim, mask=self.mask)
        self.mixed_attention = MultiHeadAttention(num_heads=num_heads, dk=dk, dv=dv, model_dim=model_dim)
        self.layerNorm1 = nn.LayerNorm(normalized_shape=model_dim)
        self.layerNorm2 = nn.LayerNorm(normalized_shape=model_dim)
        self.layerNorm3 = nn.LayerNorm(normalized_shape=model_dim)

        self.ff = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=model_dim),
        )

    def forward(self, x, x_encoder):
        attention = self.masked_attention(x)
        x = self.layerNorm1(x + attention)
        attention = self.mixed_attention(x, x_encoder)
        x = self.layerNorm2(x + attention)
        feedforward = self.ff(x)
        x = self.layerNorm3(x + feedforward)
        return x


class CustomSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, x_encoder):
        for module in self.modules_list:
            x = module(x, x_encoder)
        return x


class Decoder(nn.Module):
    def __init__(self, num_heads, dk, dv, d_ff, model_dim, max_length, dropout, num_decoders):
        super().__init__()
        decoders_list = [
            DecoderBlock(
                num_heads=num_heads,
                dk=dk,
                dv=dv,
                d_ff=d_ff,
                model_dim=model_dim,
                dropout=dropout,
                max_length=max_length,
            )
            for _ in range(num_decoders)
        ]
        self.decoders = CustomSequential(*decoders_list)

    def forward(self, x, x_encoder):
        x = self.decoders(x, x_encoder)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        batch_size,
        model_dim,
        max_length,
        vocab_size,
        num_out,
        num_heads,
        dv,
        dk,
        d_ff,
        dropout,
        num_encoders,
        num_decoders,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_heads=num_heads,
            dk=dk,
            dv=dv,
            d_ff=d_ff,
            model_dim=model_dim,
            dropout=dropout,
            num_encoders=num_encoders,
        )
        self.decoder = Decoder(
            num_heads=num_heads,
            dk=dk,
            dv=dv,
            d_ff=d_ff,
            model_dim=model_dim,
            dropout=dropout,
            num_decoders=num_decoders,
            max_length=max_length,
        )
        self.linear = nn.Linear(in_features=model_dim, out_features=num_out)
        self.embedding = Embedding(
            batch_size=batch_size,
            model_dim=model_dim,
            max_length=max_length,
            n_embedding=vocab_size,
        )

    def forward(self, x):
        x = self.embedding(x)
        x_encoder = self.encoder(x)
        x = self.decoder(x, x_encoder)
        x = self.linear(x)
        return x
