import torch
import numpy as np
import random
from model import Transformer
from train import TranslationDataset, Tokenizer, english_sentences, french_sentences, max_length_english, max_length_french

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


# Model Parameters
d_model = 128
max_length_encoder = max_length_english
max_length_decoder = max_length_french
vocab_size_encoder = len(english_tokenizer.vocab)
vocab_size_decoder = len(french_tokenizer.vocab)
num_out = vocab_size_decoder
num_heads = 8
dv = 16
dk = 16
d_ff = 512
dropout = 0.1
num_encoders = 4
num_decoders = 4

# Initialize model
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

def translate_sentence(sentence):
    model.eval()
    tokens = english_tokenizer.encode(sentence)
    encoder_input = torch.tensor(tokens).unsqueeze(0).to(device)
    encoder_mask = (encoder_input == english_tokenizer.pad_token_id).to(torch.float32).unsqueeze(0)

    generated_tokens, _ = model.generate(
        encoder_input, max_gen_length=max_length_french, padding_mask_encoder=encoder_mask, special_tokens_ids=french_tokenizer.special_tokens_ids
    )

    translated_sentence = french_tokenizer.decode(generated_tokens)
    return translated_sentence

if __name__ == "__main__":
    while True:
        sentence = input("Enter an English sentence (or 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
        translation = translate_sentence(sentence)
        print(f"Translated: {translation}")