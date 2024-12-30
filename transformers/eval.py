import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

# Create tokenizers and Dataset for English and French
english_tokenizer = Tokenizer(english_sentences)
french_tokenizer = Tokenizer(french_sentences)
dataset = TranslationDataset(
    english_sentences,
    french_sentences,
    english_tokenizer,
    french_tokenizer,
    max_length_french,
    max_length_english,
)

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
    criterion = nn.CrossEntropyLoss(ignore_index=english_tokenizer.pad_token_id)

    # Dataset split and DataLoader, wth the seed we ensure that this is the same split as in the train.py
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate the model on the test set
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            encoder_input, decoder_input, label, encoder_mask, decoder_mask = batch
            encoder_input, decoder_input, label, encoder_mask, decoder_mask = (
                encoder_input.to(device),
                decoder_input.to(device),
                label.to(device),
                encoder_mask.to(device),
                decoder_mask.to(device),
            )

            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = criterion(output.view(-1, vocab_size_decoder), label.view(-1))
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    # Show some example translations
    num_examples = 5
    for i, batch in enumerate(test_loader):
        if i >= num_examples:
            break
        encoder_input, decoder_input, label, encoder_mask, decoder_mask = batch
        encoder_input, decoder_input, label, encoder_mask, decoder_mask = (
            encoder_input.to(device),
            decoder_input.to(device),
            label.to(device),
            encoder_mask.to(device),
            decoder_mask.to(device),
        )
        generated_tokens, _ = model.generate(
            encoder_input, max_gen_length=max_length_french, padding_mask_encoder=encoder_mask, special_tokens_ids=french_tokenizer.special_tokens_ids
        )
            
        original_sentence = english_tokenizer.decode(encoder_input[0].tolist())
        translated_sentence = french_tokenizer.decode(generated_tokens)
        target_sentence = french_tokenizer.decode(label[0].tolist())

        print(f"Original: {original_sentence}")
        print(f"Translated: {translated_sentence}")
        print(f"Target: {target_sentence}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and generate translations from the test dataset using a trained Transformer model.")
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
