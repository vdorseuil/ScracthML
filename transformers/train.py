import torch
import numpy as np
import torch.nn as nn

import matplotlib.pyplot as plt

from model import Transformer


batch_size = 16
model_dim = 512
max_length = 100
vocab_size = 32000
num_out = vocab_size
num_heads = 8
dv = 64
dk = 64
d_ff = 2048
dropout = 0.1
num_encoders = 6
num_decoders = 6

x = torch.randint(0, vocab_size, (batch_size, max_length))

MyTransformer = Transformer(
    batch_size=batch_size,
    model_dim=model_dim,
    max_length=max_length,
    vocab_size=vocab_size,
    num_out=num_out,
    num_heads=num_heads,
    dv=dv,
    dk=dk,
    d_ff=d_ff,
    dropout=dropout,
    num_encoders=num_encoders,
    num_decoders=num_decoders,
)



if __name__ == "__main__":
    out = torch.softmax(MyTransformer(x), dim=-1)
    print(out.shape)
