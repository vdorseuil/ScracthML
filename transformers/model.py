import numpy as np
import torch
import torch.nn as nn

"""
Implementation of a Transformer model for sequence-to-sequence tasks, coded from scratch using the Pytorch library.

This module includes the following components:
- PositionalEncoding: Adds positional information to the input embeddings.
- Embedding: Combines token embeddings with positional encodings and applies dropout.
- SingleHeadAttention: Implements single-head attention mechanism.
- MultiHeadAttention: Implements multi-head attention mechanism using multiple single-head attentions.
- SublayerConnection: Implements dropout, add, and normalization for sublayers.
- EncoderBlock: Defines a single encoder block with multi-head attention and feed-forward layers.
- Encoder: Stacks multiple encoder blocks.
- DecoderBlock: Defines a single decoder block with masked and mixed multi-head attention and feed-forward layers.
- Decoder: Stacks multiple decoder blocks.
- Transformer: Combines the encoder and decoder to form the complete Transformer model.

We use the following sizes for the tensors of this model/
- h*dv = model_dim
- x : (batch_size, max_length) 
- tokens_id in x between 0 and vocab_size
- Embedd(x) : (batch_size, max_length, model_dim)
- K : (model_dim, dk)
- Kx : (batch_size, max_length, dk)
- Q : (model_dim, dk)
- Qx : (batch_size, max_length, dk)
- Qx*Kx^T : (batch_size, max_length, max_length)
- V : (model_dim, dv)
- Vx : (batch_size, max_length, dv)

For the mixed attention: (can have ≠ max_lengths for encoder and decoder)
- Kx : (batch_size, max_length_encoder, dk)
- Qx : (batch_size, max_length_decoder, dk)
- Qx*Kx^T : (batch_size, max_length_decoder, max_length_encoder)
- Vx : (batch_size, max_length_encoder, dv)
- SingleHeadAttention : (batch_size, max_length_decoder, dv)
- MultiHeadAttention : (batch_size, max_length_decoder, h*dv)

"""


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding layer that adds positional information to the input embeddings.

        d_model (int): The dimension of the model.
        max_length (int): The maximum length of the input sequences.

    Methods:
        SinPos(i: int, pos: int) -> float:
            Computes the sine or cosine positional encoding for a given position and dimension index.

        compute() -> None:
            Computes the positional encoding matrix for all positions up to max_length.

        forward(x: torch.Tensor) -> torch.Tensor:
            Adds the positional encoding to the input tensor.
    """

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
        return self.Mat[: x.shape[-1], :]


class Embedding(nn.Module):
    """
    Embedding layer with positional encoding and dropout.

        d_model (int): The dimension of the embedding vector.
        max_length (int): The maximum length of the input sequences.
        n_embedding (int): The number of embeddings (vocabulary size).
        dropout (float): The dropout rate to apply after adding positional encoding.

    Attributes:
        max_length (int): The maximum length of the input sequences.
        d_model (int): The dimension of the embedding vector.
        n_embedding (int): The number of embeddings (vocabulary size).
        embedding (nn.Embedding): The embedding layer.
        pos_encoding (PositionalEncoding): The positional encoding layer.
        dropout (nn.Dropout): The dropout layer.

    Methods:
        forward(x):
            Applies embedding, positional encoding, and dropout to the input tensor.

                x (torch.Tensor): The input tensor of token indices.

            Returns:
                torch.Tensor: The output tensor after embedding, positional encoding, and dropout.
    """

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
        device = x.device

        x = self.embedding(x) + self.pos_encoding(x).to(device)
        return self.dropout(x)


class SingleHeadAttention(nn.Module):
    """
    SingleHeadAttention module implementing a single head attention mechanism.

        dk (int): Dimensionality of the key vectors.
        dv (int): Dimensionality of the value vectors.
        d_model (int): Dimensionality of the input feature vectors.

    Attributes:
        dk (int): Dimensionality of the key vectors.
        dv (int): Dimensionality of the value vectors.
        d_model (int): Dimensionality of the input feature vectors.
        K (nn.Linear): Linear layer to project input to key vectors.
        Q (nn.Linear): Linear layer to project input to query vectors.
        V (nn.Linear): Linear layer to project input to value vectors.

    Methods:
        forward(x: torch.Tensor, x_encoder: torch.Tensor = None, mask=None) -> torch.Tensor:
            Computes the attention output for the given input tensors.

                x (torch.Tensor): Input tensor of shape (batch_size, max_length, d_model).
                x_encoder (torch.Tensor, optional): Encoder output tensor of shape (batch_size, length, d_model). Defaults to None.
                mask (torch.Tensor, optional): Mask tensor to apply to the attention scores. Defaults to None.

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, max_length, dv) after applying attention mechanism.
    """

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
    """
    MultiHeadAttention module implementing the Multi Head Attention mechanism using the class SingleHeadAttention.

        num_heads (int): Number of attention heads.
        dk (int): Dimension of the keys.
        dv (int): Dimension of the values.
        d_model (int): Dimension of the model (output dimension).

    Attributes:
        attention_heads (nn.ModuleList): List of single head attention modules.
        WO (nn.Linear): Linear layer to project concatenated outputs of attention heads.

    Methods:
        forward(x: torch.Tensor, x_encoder: torch.Tensor = None, mask=None) -> torch.Tensor:
            Computes the multi-head attention output.

                x (torch.Tensor): Input tensor.
                x_encoder (torch.Tensor, optional): Encoder input tensor for cross-attention. Defaults to None.
                mask (optional): Masking tensor to prevent attention to certain positions. Defaults to None.

            Returns:
                torch.Tensor: Output tensor after applying multi-head attention and linear projection.
    """

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


class SublayerConnection(nn.Module):
    """
    SublayerConnection applies dropout, addition, and normalization to the input tensor.

    Args:
        features (int): The number of features in the input tensor.
        dropout (float): The dropout probability.
        eps (float, optional): A small value to avoid division by zero in normalization. Default is 1e-6.

    Methods:
        forward(x, sublayer_output):
            Applies dropout to the sublayer output, adds it to the input tensor, and normalizes the result.

            Args:
                x (torch.Tensor): The input tensor.
                sublayer_output (torch.Tensor): The output tensor from the sublayer.

            Returns:
                torch.Tensor: The normalized tensor after applying dropout and addition.

    """

    def __init__(self, features, dropout, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.dropout = nn.Dropout(p=dropout)

        self.eps = eps

    def forward(self, x, sublayer_output):
        x = x + self.dropout(sublayer_output)  # dropout and add
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # norm
        return x


class EncoderBlock(nn.Module):
    """
    Implementation of a single encoder block module.

    Args:
        num_heads (int): Number of attention heads.
        dk (int): Dimension of the key vectors.
        dv (int): Dimension of the value vectors.
        d_ff (int): Dimension of the feedforward network.
        d_model (int): Dimension of the model.
        dropout (float): Dropout rate.

    Attributes:
        attention (MultiHeadAttention): Multi-head attention mechanism.
        sublayer1 (SublayerConnection): First sublayer connection.
        sublayer2 (SublayerConnection): Second sublayer connection.
        ff (nn.Sequential): Feedforward neural network.

    Methods:
        forward(x, mask=None):
            Forward pass through the encoder block.

            Args:
                x (torch.Tensor): Input tensor.
                mask (torch.Tensor, optional): Mask tensor. Defaults to None.

            Returns:
                torch.Tensor: Output tensor after processing through the encoder block.
    """

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
    """
    Implementation of the encoder module using several encoder blocks.

    Args:
        num_heads (int): Number of attention heads.
        dk (int): Dimension of the key vectors.
        dv (int): Dimension of the value vectors.
        d_ff (int): Dimension of the feed-forward layer.
        d_model (int): Dimension of the model.
        dropout (float): Dropout rate.
        num_encoders (int): Number of encoder blocks.

    Attributes:
        encoders_list (list): List of encoder blocks.
        encoders (nn.ModuleList): ModuleList containing the encoder blocks.

    Methods:
        forward(x, mask=None):
            Passes the input through each encoder block in the encoder stack.

            Args:
                x (torch.Tensor): Input tensor.
                mask (torch.Tensor, optional): Mask tensor. Defaults to None.

            Returns:
                torch.Tensor: Output tensor after passing through the encoder stack.
    """

    def __init__(self, num_heads, dk, dv, d_ff, d_model, dropout, num_encoders):
        super().__init__()
        self.encoders_list = [
            EncoderBlock(
                num_heads=num_heads,
                dk=dk,
                dv=dv,
                d_ff=d_ff,
                d_model=d_model,
                dropout=dropout,
            )
            for _ in range(num_encoders)
        ]
        self.encoders = nn.ModuleList(self.encoders_list)

    def forward(self, x, mask=None):
        for encoder in self.encoders_list:
            x = encoder(x, mask)
        return x


class DecoderBlock(nn.Module):
    """
    Implementation of a single decoder block module.

    Args:
        num_heads (int): Number of attention heads.
        dk (int): Dimension of the key vectors.
        dv (int): Dimension of the value vectors.
        d_ff (int): Dimension of the feedforward network.
        d_model (int): Dimension of the model.
        dropout (float): Dropout rate.

    Attributes:
        masked_attention (MultiHeadAttention): Multi-head self-attention mechanism with masking.
        mixed_attention (MultiHeadAttention): Multi-head attention mechanism for encoder-decoder attention.
        sublayer1 (SublayerConnection): Sublayer connection for masked attention.
        sublayer2 (SublayerConnection): Sublayer connection for mixed attention.
        sublayer3 (SublayerConnection): Sublayer connection for feedforward network.
        ff (nn.Sequential): Feedforward neural network.

    Methods:
        forward(x, x_encoder, causal_mask=None, mixed_mask=None):
            Forward pass through the decoder block.

            Args:
                x (torch.Tensor): Input tensor.
                x_encoder (torch.Tensor): Encoder output tensor.
                causal_mask (torch.Tensor, optional): Mask for the self-attention mechanism.
                mixed_mask (torch.Tensor, optional): Mask for the encoder-decoder attention mechanism.

            Returns:
                torch.Tensor: Output tensor after passing through the decoder block.
    """

    """"""

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
    """
    Implementation of the decoder module using several decoder blocks.

    Args:
        num_heads (int): Number of attention heads.
        dk (int): Dimension of the key vectors.
        dv (int): Dimension of the value vectors.
        d_ff (int): Dimension of the feed-forward layer.
        d_model (int): Dimension of the model.
        dropout (float): Dropout rate.
        num_decoders (int): Number of decoder blocks.

    Attributes:
        decoders (nn.ModuleList): List of decoder blocks.

    Methods:
        forward(x, x_encoder, causal_mask=None, mixed_mask=None):
            Passes the input through the stack of decoder blocks.

            Args:
                x (torch.Tensor): Input tensor.
                x_encoder (torch.Tensor): Encoder output tensor.
                causal_mask (torch.Tensor, optional): Causal mask tensor.
                mixed_mask (torch.Tensor, optional): Mixed mask tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the decoder blocks.
    """

    def __init__(self, num_heads, dk, dv, d_ff, d_model, dropout, num_decoders):
        super().__init__()
        decoders_list = [
            DecoderBlock(
                num_heads=num_heads,
                dk=dk,
                dv=dv,
                d_ff=d_ff,
                d_model=d_model,
                dropout=dropout,
            )
            for _ in range(num_decoders)
        ]
        self.decoders = nn.ModuleList(decoders_list)

    def forward(self, x, x_encoder, causal_mask=None, mixed_mask=None):
        for decoder in self.decoders:
            x = decoder(x, x_encoder, causal_mask, mixed_mask)
        return x


class Transformer(nn.Module):
    """
    Implemenation of the Transformer model for sequence-to-sequence tasks.

        d_model (int) : Dimension of the model.
        max_length_encoder (iut): Maximum length of the encoderinput sequence.
        vocab_size_encoder (int): Vocabulary size of the encoder.
        max_length_decoder (int): Maximum length of the decoder input sequence.
        vocab_size_decoder (int): Vocabulary size of the decoder.
        num_out (int): Number of output features.
        num_heads (int): Number of attention heads.
        dk (int): Dimensionality of the key/query vectors.
        d_ff (int): Dimensionality of the feed-forward network.
        dropout (float): Dropout rate.
        num_encoders (int): Number of encoder layers.
        num_decoders (int): Number of decoder layers.

    Methods:
        forward(input, output, padding_mask_encoder, padding_mask_decoder):
            Forward pass of the Transformer model.
            Args:
                input (torch.Tensor): Input tensor for the encoder.
                output (torch.Tensor): Input tensor for the decoder.
                padding_mask_encoder (torch.Tensor): Padding mask for the encoder.
                padding_mask_decoder (torch.Tensor): Padding mask for the decoder.
            Returns:
                torch.Tensor: Output tensor of the model.

        generate(input, max_gen_length, start_token, end_token, padding_mask_encoder):
            Generate a sequence using greedy decoding.
            Args:
                input (torch.Tensor): Input tensor for the encoder.
                max_gen_length (int): Maximum length of the generated sequence.
                start_token (int): Token to start the generation.
                end_token (int): Token to end the generation.
                padding_mask_encoder (torch.Tensor): Padding mask for the encoder.
            Returns:
                tuple: Generated tokens and their probabilities.
    """

    def __init__(
        self,
        d_model,
        max_length_encoder,
        vocab_size_encoder,
        max_length_decoder,
        vocab_size_decoder,
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
        self.embedding_encoder = Embedding(d_model, max_length_encoder, n_embedding=vocab_size_encoder, dropout=dropout)
        self.embedding_decoder = Embedding(d_model, max_length_decoder, n_embedding=vocab_size_decoder, dropout=dropout)
        self.encoder = Encoder(num_heads, dk, dv, d_ff, d_model, dropout, num_encoders)
        self.decoder = Decoder(num_heads, dk, dv, d_ff, d_model, dropout, num_decoders)
        self.linear = nn.Linear(in_features=d_model, out_features=num_out)
        self.ff_mask = torch.zeros(max_length_decoder, max_length_decoder) + torch.triu(
            torch.full((max_length_decoder, max_length_decoder), float(-1e9)),
            diagonal=1,
        )
        self.max_length_encoder = max_length_encoder
        self.max_length_decoder = max_length_decoder

    def forward(self, input, output, padding_mask_encoder, padding_mask_decoder):
        device = input.device

        padding_mask_decoder[padding_mask_decoder == 1] = -1e9
        padding_mask_decoder[padding_mask_decoder == 0] = 0.0

        padding_mask_encoder[padding_mask_encoder == 1] = -1e9
        padding_mask_encoder[padding_mask_encoder == 0] = 0.0

        encoder_mask = padding_mask_encoder.unsqueeze(1).expand(-1, self.max_length_encoder, -1)

        decoder_mask = self.ff_mask.to(device) + padding_mask_decoder.unsqueeze(1).expand(-1, self.max_length_decoder, -1)  # Causal mask
        mixed_mask = padding_mask_decoder.unsqueeze(1).expand(-1, self.max_length_encoder, -1).transpose(-1, -2) + padding_mask_encoder.unsqueeze(
            1
        ).expand(-1, self.max_length_decoder, -1)
        # In reality we don't care about the padded rows of our attention matrix, at the end the loss won't take them into account and we won't update the weights during the backward pass.

        input_embed = self.embedding_encoder(input)
        output_embed = self.embedding_decoder(output)

        x_encoder = self.encoder(input_embed, mask=encoder_mask)
        x = self.decoder(output_embed, x_encoder, causal_mask=decoder_mask, mixed_mask=mixed_mask)
        x = self.linear(x)
        return x

    def generate(self, input, max_gen_length, padding_mask_encoder, special_tokens_ids):  # greedy decoding
        self.eval()
        input_embed = self.embedding_encoder(input)

        padding_mask_encoder[padding_mask_encoder == 1] = -1e9
        padding_mask_encoder[padding_mask_encoder == 0] = 0.0
        x_encoder = self.encoder(input_embed, mask=padding_mask_encoder.to(input.device))

        generated_tokens = [special_tokens_ids["bos_token_id"]]
        generated_tokens_probas = [1]

        for _ in range(max_gen_length):
            output = torch.tensor(generated_tokens).unsqueeze(0).to(input.device)
            out_embed = self.embedding_decoder(output)
            causal_mask = self.ff_mask[: out_embed.size(1), : out_embed.size(1)].to(input.device)
            x = self.decoder(out_embed, x_encoder, causal_mask=causal_mask, mixed_mask=padding_mask_encoder)
            x = self.linear(x)
            probas = torch.softmax(x, dim=-1)

            probas[:, :, special_tokens_ids["unk_token_id"]] = 0  # Mask the probability of the <UNK> token
            probas[:, :, special_tokens_ids["pad_token_id"]] = 0  # Mask the probability of the <PAD> token

            max_proba, next_token = torch.max(probas[:, -1, :], dim=-1)  # greedy decoding : only max_proba
            generated_tokens.append(next_token.item())
            generated_tokens_probas.append(max_proba.item())
            if next_token == special_tokens_ids["eos_token_id"]:
                break

        return generated_tokens, generated_tokens_probas
