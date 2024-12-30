import argparse
import random

import numpy as np
import torch

from model import Transformer
from train import (Tokenizer, TranslationDataset, english_sentences,
                   french_sentences, max_length_english, max_length_french)

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Create tokenizers
english_tokenizer = Tokenizer(english_sentences)
french_tokenizer = Tokenizer(french_sentences)


def translate_sentence(model, sentence, device):
    model.eval()
    tokens = english_tokenizer.encode(sentence)
    encoder_input = torch.tensor(tokens).unsqueeze(0).to(device)
    encoder_mask = (encoder_input == english_tokenizer.pad_token_id).to(torch.float32).unsqueeze(0)

    generated_tokens, _ = model.generate(
        encoder_input, max_gen_length=max_length_french, padding_mask_encoder=encoder_mask, special_tokens_ids=french_tokenizer.special_tokens_ids
    )

    translated_sentence = french_tokenizer.decode(generated_tokens)
    return translated_sentence


def main(args):
    # Model Parameters
    d_model = args.d_model
    max_length_encoder = max_length_english
    max_length_decoder = max_length_french
    vocab_size_encoder = len(english_tokenizer.vocab)
    vocab_size_decoder = len(french_tokenizer.vocab)
    num_out = vocab_size_decoder
    num_heads = args.num_heads
    dv = args.dv
    dk = args.dk
    d_ff = args.d_ff
    dropout = args.dropout
    num_encoders = args.num_encoders
    num_decoders = args.num_decoders

    # Initialize model with the same parameters as your checkpoint
    model = Transformer(
        d_model=d_model,
        max_length_encoder=max_length_encoder,
        max_length_decoder=max_length_decoder,
        vocab_size_encoder=vocab_size_encoder,
        vocab_size_decoder=vocab_size_decoder,
        num_out=num_out,
        num_heads=num_heads,
        dv=dv,
        dk=dk,
        d_ff=d_ff,
        dropout=dropout,
        num_encoders=num_encoders,
        num_decoders=num_decoders,
    )

    # Load the best model
    model.load_state_dict(torch.load("best_transformer_model.pth", weights_only=True))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    while True:
        sentence = input("Enter an English sentence (or 'quit' to exit): ")
        if sentence.lower() == "quit":
            break
        translation = translate_sentence(model, sentence, device)
        print(f"Translated: {translation}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate custom English to French translations using a trained Transformer model.")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dv", type=int, default=16, help="Dimension of value vectors")
    parser.add_argument("--dk", type=int, default=16, help="Dimension of key/query vectors")
    parser.add_argument("--d_ff", type=int, default=512, help="Dimension of feedforward layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_encoders", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num_decoders", type=int, default=4, help="Number of decoder layers")

    args = parser.parse_args()

    main(args)
