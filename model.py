import numpy as np
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, batch_size, model_dim, max_length, n_embedding):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_dim = model_dim
        self.n_embedding = n_embedding
        self.embedding = torch.nn.Embedding(
            num_embeddings=n_embedding, embedding_dim=model_dim
        )
        self.pos_encoding = PositionalEncoding(
            batch_size=batch_size, model_dim=model_dim, max_length=max_length
        )
        pass

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding(x)
        return x


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
        Mat = torch.Tensor(
            [
                [self.SinPos(i, pos) for i in range(self.model_dim)]
                for pos in range(self.max_length)
            ]
        )
        self.Mat = Mat.expand(self.batch_size, -1, -1)

    def forward(self, x):
        with torch.no_grad():
            return self.Mat


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
        if x_encoder is not None:
            Kx = self.K(x_encoder)
            Vx = self.V(x_encoder)
        else:
            Kx = self.K(x)
            Vx = self.V(x)
        Qx = self.Q(x)
        QK = torch.matmul(Qx, Kx.transpose(-2, -1)) / np.sqrt(self.dk)
        if self.mask is not None:
            QK += self.mask
        QK = torch.softmax(QK, dim=-1)
        return torch.matmul(QK, Vx)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dk: int, dv: int, model_dim: int, mask=None):
        super().__init__()
        assert (
            num_heads * dv == model_dim
        ), "num_heads * dv should be equal to the model dim"
        self.attention_heads = nn.ModuleList(
            [SingleHeadAttention(dk, dv, model_dim, mask) for _ in range(num_heads)]
        )
        self.WO = nn.Linear(in_features=num_heads * dv, out_features=model_dim)
        self.mask = mask

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor = None):
        outputs = []
        for head in self.attention_heads:
            outputs.append(head(x, x_encoder))
        x = torch.cat(outputs, dim=-1)
        x = self.WO(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dk, dv, model_dim):
        super().__init__()
        self.attention = MultiHeadAttention(
            num_heads=num_heads, dk=dk, dv=dv, model_dim=model_dim
        )
        self.layerNorm = nn.LayerNorm(normalized_shape=model_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=model_dim),
        )

    def forward(self, x):
        attention = self.attention(x)
        x = self.layerNorm(x + attention)
        feedforward = self.ff(x)
        x = self.layerNorm(x + feedforward)
        return x


class Encoder(nn.Module):
    def __init__(self, num_heads, dk, dv, model_dim, num_encoders):
        super().__init__()
        self.encoders_list = [
            EncoderBlock(num_heads, dk, dv, model_dim) for _ in range(num_encoders)
        ]
        self.encoders = nn.Sequential(*self.encoders_list)

    def forward(self, x):
        x = self.encoders(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, dk, dv, model_dim, max_length):
        super().__init__()
        self.mask = torch.zeros(max_length, max_length) + torch.triu(
            torch.full((max_length, max_length), float("-inf")), diagonal=1
        )
        self.masked_attention = MultiHeadAttention(
            num_heads=num_heads, dk=dk, dv=dv, model_dim=model_dim, mask=self.mask
        )
        self.mixed_attention = MultiHeadAttention(
            num_heads=num_heads, dk=dk, dv=dv, model_dim=model_dim
        )
        self.layerNorm = nn.LayerNorm(normalized_shape=model_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=model_dim),
        )

    def forward(self, x, x_encoder):
        attention = self.masked_attention(x)
        x = self.layerNorm(x + attention)
        attention = self.mixed_attention(x, x_encoder)
        x = self.layerNorm(x + attention)
        feedforward = self.ff(x)
        x = self.layerNorm(x + feedforward)
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
    def __init__(self, num_heads, dk, dv, model_dim, max_length, num_decoders):
        super().__init__()
        decoders_list = [
            DecoderBlock(num_heads, dk, dv, model_dim, max_length)
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
        num_encoders,
        num_decoders,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_heads=num_heads,
            dk=dk,
            dv=dv,
            model_dim=model_dim,
            num_encoders=num_encoders,
        )
        self.decoder = Decoder(
            num_heads=num_heads,
            dk=dk,
            dv=dv,
            model_dim=model_dim,
            num_decoders=num_decoders,
            max_length=max_length,
        )
        self.softmax = nn.Softmax(dim=-1)
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
        x = self.softmax(x)
        return x
