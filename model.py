import torch
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt


class Embedding(nn.Module):
    def __init__(self,  batch_size, model_dim, max_length, n_embedding):
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

class PositionalEncoding(nn.Module):
    def __init__(self, batch_size, model_dim, max_length):
        super().__init__()
        self.model_dim = model_dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.compute()

    def SinPos(self, i: int, pos: int):
        if i % 2 == 0:
            return np.sin(pos / 10000.0 ** (2 * i / self.model_dim))
        else:
            return np.cos(pos / 10000.0 ** (2 * i / self.model_dim))

    def compute(self):

        Mat = torch.Tensor([[self.SinPos(i, pos) for i in range(self.model_dim)] for pos in range(self.max_length)])
        self.Mat = Mat.expand(self.batch_size, -1, -1)

    
    def forward(self, x):
        with torch.no_grad():
            return self.Mat



class SingleHeadAttention(nn.Module):
    def __init__(self, dk:int, dv:int, model_dim:int):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.model_dim = model_dim
        self.K = nn.Linear(in_features=model_dim, out_features=dk)
        self.Q = nn.Linear(in_features=model_dim, out_features=dk)
        self.V = nn.Linear(in_features=model_dim, out_features=dv)

    def forward(self, x:torch.Tensor):
        Kx = self.K(x)
        Qx = self.Q(x)
        Vx = self.V(x)
        QK = torch.sum(Kx *Qx, dim=-1)/np.sqrt(self.dk)
        QK = torch.softmax(QK, dim=-1)
        QK = QK.unsqueeze(-1)
        QK = QK.expand(-1, -1, self.dv)
        return QK*Vx
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads : int, dk:int, dv:int, model_dim:int):
        super().__init__()
        assert num_heads * dv == model_dim, "num_heads * dv should be equal to the model dim"
        self.attention_heads = nn.ModuleList([SingleHeadAttention(dk, dv, model_dim) for _ in range(num_heads)])
        self.WO = nn.Linear(in_features=num_heads*dv, out_features=model_dim)  
    
    def forward(self, x:torch.Tensor):
        outputs = [head(x) for head in self.attention_heads]
        x = torch.cat(outputs, dim=-1)
        x = self.WO(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dk, dv, model_dim):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, dk=dk, dv=dv, model_dim=model_dim)
        self.layerNorm = nn.LayerNorm(normalized_shape=model_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=model_dim)
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
        encoders_list = [EncoderBlock(num_heads, dk, dv, model_dim) for _ in range(num_encoders)]
        self.encoders = nn.Sequential(*encoders_list)

    def forward(self, x):
        x = self.encoders(x)
        return x