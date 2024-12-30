import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


import zipfile

from model import Transformer

# Read the file and Prepare the dataset

filename = "fra-eng.zip"
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(".")

with open("fra.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

pairs = [line.strip().split("\t") for line in lines]
english_sentences = [pair[0] for pair in pairs]
french_sentences = [pair[1] for pair in pairs]


# Tokenizer class
class Tokenizer:
    """
    A simple tokenizer class to encode and decode sentences into token indices and vice versa.
    We don't use any algorithm (BPE, WordPiece, SentencePiece,...) for simplicity.
        sentences (list): List of sentences.

    Attributes:
        bos_token (str): Beginning of sentence token.
        eos_token (str): End of sentence token.
        pad_token (str): Padding token.
        special_tokens (list): List of special tokens.
        vocab (list): List of vocabulary words including special tokens.
        word_to_idx (dict): Dictionary mapping words to their corresponding indices.
        idx_to_word (dict): Dictionary mapping indices to their corresponding words.
        sos_token_id (int): Index of the beginning of sentence token.
        eos_token_id (int): Index of the end of sentence token.
        pad_token_id (int): Index of the padding token.

    Methods:
        __init__(sentences):
            Initializes the tokenizer with a list of sentences and builds the vocabulary.
        build_vocab(sentences):
            Builds the vocabulary from a list of sentences.
        encode(sentence):
            Encodes a sentence into a list of token indices.
        decode(indices):
            Decodes a list of token indices into a sentence.
    """

    def __init__(self, sentences):
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.special_tokens = [self.bos_token, self.eos_token, self.pad_token]
        self.build_vocab(sentences)

    def build_vocab(self, sentences):
        self.vocab = set()
        for sentence in sentences:
            self.vocab.update(sentence.split())
        self.vocab = sorted(list(self.vocab))
        self.vocab = self.special_tokens + self.vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        self.sos_token_id = self.word_to_idx[self.bos_token]
        self.eos_token_id = self.word_to_idx[self.eos_token]
        self.pad_token_id = self.word_to_idx[self.pad_token]

    def encode(self, sentence):
        return [self.word_to_idx[word] for word in sentence.split()]

    def decode(self, indices):
        return " ".join(
            [
                self.idx_to_word[idx]
                for idx in indices
                if idx not in {self.sos_token_id, self.eos_token_id, self.pad_token_id}
            ]
        )


# Dataset class
class TranslationDataset(Dataset):
    """
    English to French translation Dataset
        english_sentences (list of str): List of English sentences.
        french_sentences (list of str): List of French sentences.
        english_tokenizer (Tokenizer): Tokenizer for English sentences.
        french_tokenizer (Tokenizer): Tokenizer for French sentences.
        max_length_french (int): Maximum length for French sentences.
        max_length_english (int): Maximum length for English sentences.

    Attributes:
        english_sentences (list of str): List of English sentences.
        french_sentences (list of str): List of French sentences.
        english_tokenizer (Tokenizer): Tokenizer for English sentences.
        french_tokenizer (Tokenizer): Tokenizer for French sentences.
        max_length_french (int): Maximum length for French sentences.
        max_length_english (int): Maximum length for English sentences.
        data (list of dict): Preprocessed data containing encoder inputs, decoder inputs, labels, and masks.

    Methods:
        preprocess(): Preprocesses the input sentences and returns a list of dictionaries containing encoder inputs, decoder inputs, labels, and masks.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the encoder input, decoder input, label, encoder mask, and decoder mask for the given index.

    Comments: (sizes)
        - Encoder Input : [tokenA, tokenB, tokenC] #English
        - Target Sequence: [BOS, token1, token2, token3, EOS] #French
        - Decoder Input: [BOS, token1, token2, token3] #French
        - Labels: [token1, token2, token3, EOS] # French
    """

    def __init__(
        self,
        english_sentences,
        french_sentences,
        english_tokenizer,
        french_tokenizer,
        max_length_french,
        max_length_english,
    ):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.english_tokenizer = english_tokenizer
        self.french_tokenizer = french_tokenizer
        self.max_length_french = max_length_french
        self.max_length_english = max_length_english
        self.data = self.preprocess()

    def preprocess(self):  #
        data = []
        for eng, fra in zip(self.english_sentences, self.french_sentences):
            if (
                len(eng.split()) > self.max_length_english
                or len(fra.split()) > self.max_length_french - 1
            ):
                continue  # We remove the sequences that are too long (only 800 and 150 of each -> not a lot)

            eng_tokens = eng.split()
            fra_tokens = (
                [self.french_tokenizer.bos_token]
                + fra.split()
                + [self.french_tokenizer.eos_token]
            )

            eng_idx = self.english_tokenizer.encode(" ".join(eng_tokens))
            fra_idx = self.french_tokenizer.encode(" ".join(fra_tokens))

            eng_idx += [self.english_tokenizer.pad_token_id] * (
                self.max_length_english - len(eng_idx)
            )
            fra_idx += [self.french_tokenizer.pad_token_id] * (
                self.max_length_french - len(fra_idx) + 1
            )  # +1 because then we will shift outputs to the right

            encoder_input = torch.tensor(eng_idx)
            decoder_input = torch.tensor(fra_idx[:-1])
            label = torch.tensor(fra_idx[1:])

            encoder_mask = (encoder_input == self.english_tokenizer.pad_token_id).to(
                torch.float32
            )

            decoder_mask = (decoder_input == self.french_tokenizer.pad_token_id).to(
                torch.float32
            )

            data.append(
                {
                    "encoder_input": encoder_input,
                    "decoder_input": decoder_input,
                    "label": label,
                    "encoder_mask": encoder_mask,
                    "decoder_mask": decoder_mask,
                }
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["encoder_input"],
            item["decoder_input"],
            item["label"],
            item["encoder_mask"],
            item["decoder_mask"],
        )


# Create tokenizers and Datset for English and French
max_length_english = 20  # We choose thes values specifically for this dataset.
max_length_french = 25
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

# Training Parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion and optimizer
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

criterion = nn.CrossEntropyLoss(ignore_index=english_tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train(num_epochs):
    """Train our transformer model for a certain number of epochs.

    Args:
        num_epochs (int): Number of epochs

    Returns:
        train losses and validation losses.
    """
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            encoder_input, decoder_input, label, encoder_mask, decoder_mask = batch
            encoder_input, decoder_input, label, encoder_mask, decoder_mask = (
                encoder_input.to(device),
                decoder_input.to(device),
                label.to(device),
                encoder_mask.to(device),
                decoder_mask.to(device),
            )

            optimizer.zero_grad()
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = criterion(output.view(-1, vocab_size_decoder), label.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
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
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print(f"Model saved with validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses


if __name__ == "__main__":
    print(
        "Training of a Transformer Model on the English-French Dataset with the following hyperparameters:"
    )
    print(
        f"Model Parameters: d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}, num_encoders={num_encoders}, num_decoders={num_decoders}"
    )
    print(
        f"Training Parameters: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}"
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the Transformer model: {num_params}")

    train_losses, val_losses = train(num_epochs)

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("loss_plot.png")
    plt.show()
