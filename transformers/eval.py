import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Transformer
from train import TranslationDataset, Tokenizer, english_sentences, french_sentences, max_length_english, max_length_french

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

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
_, _, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# Create DataLoader for the test set
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
model.load_state_dict(torch.load("best_transformer_model.pth"))
model.eval()

# Model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Criterion for computing test loss
criterion = nn.CrossEntropyLoss(ignore_index=english_tokenizer.pad_token_id)

# # Evaluate the model on the test set
# total_test_loss = 0
# with torch.no_grad():
#     for batch in test_loader:
#         encoder_input, decoder_input, label, encoder_mask, decoder_mask = batch
#         encoder_input, decoder_input, label, encoder_mask, decoder_mask = (
#             encoder_input.to(device),
#             decoder_input.to(device),
#             label.to(device),
#             encoder_mask.to(device),
#             decoder_mask.to(device),
#         )

#         output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
#         loss = criterion(output.view(-1, vocab_size_decoder), label.view(-1))
#         total_test_loss += loss.item()

# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Average Test Loss: {avg_test_loss:.4f}")

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
        encoder_input, max_gen_length=max_length_french, start_token=french_tokenizer.sos_token_id, end_token=french_tokenizer.eos_token_id, padding_mask_encoder=encoder_mask
    )

    original_sentence = english_tokenizer.decode(encoder_input[0].tolist())
    translated_sentence = french_tokenizer.decode(generated_tokens)
    target_sentence = french_tokenizer.decode(label[0].tolist())

    print(f"Original: {original_sentence}")
    print(f"Translated: {translated_sentence}")
    print(f"Target: {target_sentence}")
    print()