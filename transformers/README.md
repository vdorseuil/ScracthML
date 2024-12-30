# Transformers

This folder contains an implementation of the Transformer architecture based on the original paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

## Files

- **[`model.py`](model.py)**: Contains the implementation of the Transformer model, including the encoder, decoder, and self-attention mechanisms.
- **[`train.py`](train.py)**: Script to train the Transformer model on a dataset, including data loading, training loop, and checkpoint saving.
- **[`eval.py`](eval.py)**: Script to evaluate the Transformer model on the test set, including generating example translations and computing the test loss.
- **[`generate.py`](/transformers/generate.py)**: Script to interactively translate English sentences to French using the trained Transformer model.

## Highlights

- This implementation is coded entirely from scratch based on the original paper.
- No pre-built libraries (e.g., Hugging Face) were used for the core architecture. We only used Pytorch and Numpy.
- The goal was to deeply understand the workings of the Transformer model by building it myself.

## Features Implemented
- Multi Head Attention
- Handles different max_length for encoder and decoder
- Regularization (Dropout and normalization)
- Forward and generate function for the model
- Causal mask, Padding Mask
- Tokenizer from scratch with BOS, EOS, UNK and PAD tokens
- Dataset preprocessing for training

## How to Use : Training

1. Ensure you have the required dependencies installed (e.g., PyTorch, NumPy).
2. Download the `fra-eng.zip` file from [ManyThings.org](http://www.manythings.org/anki/)
3. Run [`train.py`](train.py) to train the model:

   ```bash
   python train.py
   ```
4. To customize hyperparameters, you can specify them as command-line arguments:
   ```bash 
   python train.py --d_model 256 --num_heads 8 --dv 32 --dk 32 --d_ff 1024 --dropout 0.2 --num_encoders 6 --num_decoders 6 --batch_size 32 --learning_rate 0.0005 --num_epochs 20
   ```

Feel free to explore the code to understand the implementation and adapt it to your needs!


## Evaluation
To evaluate the model on the test set and generate example translations, run the [`eval.py`](eval.py) script:

   ```bash
   python eval.py 
   ```
Or if you didn't use default hyperparameters : 

   ```bash
   python eval.py --d_model 256 --num_heads 8 --dv 32 --dk 32 --d_ff 1024 --dropout 0.2 --num_encoders 6 --num_decoders 6
   ```

The [`eval.py`](eval.py) script will:

- Compute the average test loss.
- Generate and print example translations from the test set.

**Note:** Ensure that you pass the same hyperparameters as arguments to `eval.py` that you used in train.py to correctly load the model.


## Interactive Translation
To interactively translate English sentences to French using the trained Transformer model, run the [`generate.py`](generate.py) script:
   ```bash
   python generate.py
   ```
The [`generate.py`](generate.py) script will:

- Load the trained Transformer model.
- Prompt you to enter English sentences.
- Output the translated French sentences.

**Note:** Same note for the generate.py script, ensure that you pass the same hyperparameters as aruments to `generate.py` that you used in `train.py`.


## Results
### Results for a 42M Parameter Model

To test our scripts, we trained a model with the following hyperparameters for 50 epochs, resulting in a total of 42 million parameters.
- Model Parameters: d_model=256, d_ff=512, num_encoders=6, num_decoders=6
num_heads=8, dv=32, dk=32, dropout=0.1
- Training Parameters: batch_size=256, learning_rate=0.001, num_epochs=50


Below are the key results and observations.

- **Average Test Loss**: 1.5633
- **Example from the test dataset:**
   - Original: Let's all remember to be nice.
      - Translated: Rappelons toute bonne d'être gentille.
      - Target: Souvenons-nous tous d'être gentils.

   - Original: Why would I be jealous?
      - Translated: Pourquoi serais-je jalouse ?
      - Target: Pourquoi serais-je jaloux ?

   - Original: I felt like I was dead.
      - Translated: J'avais l'impression d'être mort.
      - Target: J'avais l'impression d'être mort.

   - Original: The ice has melted.
      - Translated: La glace a fondu.
      - Target: La glace a fondu.
 
   - Original: You've got to read this.
      - Translated: Tu dois lire ça.
      - Target: Il faut que tu lises ça.

### Comments on Model Performance

- The model was trained for 50 epochs with a relatively small architecture due to computational limitations. It achieved best perfomance after epoch 17.
- The goal was not to achieve state-of-the-art (SOTA) performance but to understand the Transformer model's workings.
- Despite the limitations, the model has learned some aspects of translation, often getting simple sentences correct.
- We used a very basic tokenizer; implementing a BPE would definitely help.
- There are still errors, especially with longer and more complex sentences, indicating room for improvement with more training and a larger model.
- By playing with the model using `generate.py` to try different type sentences, we saw some limitations. Due to the basic implementation of the tokenizer, the model was unfortunately very sensitive to the punctuation, the spaces and the case. But for simple sentences (like the ones in the dataset), the model seems pretty good.
### Loss Plot during Training

Below is the plot of the training and validation losses for 50 epoch for this model. We monitored the best checkpoint on the validation loss to avoid overfitting. The best model obtained a val_loss of 1.5638 on epoch 17.

![Loss Plot](loss_plot.png)


## Data Source
This project uses data from the [Tatoeba Project](http://tatoeba.org) and [ManyThings.org](http://www.manythings.org/anki/).

### Attribution
The dataset used to train the model comes from the `fra-eng.zip` file.

This data is provided under the [Creative Commons Attribution 2.0 License](http://creativecommons.org/licenses/by/2.0).

### Terms of Use
Please see the terms of use for the dataset:
- [Tatoeba Terms of Use](http://tatoeba.org/eng/terms_of_use)
- [Creative Commons Attribution 2.0 License](http://creativecommons.org/licenses/by/2.0)

