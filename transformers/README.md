# Transformers

This folder contains an implementation of the Transformer architecture based on the original paper:

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

## Files

- **`model.py`**: Contains the implementation of the Transformer model, including the encoder, decoder, and self-attention mechanisms.
- **`train.py`**: Script to train the Transformer model on a dataset, including data loading, training loop, and checkpoint saving.
- **`eval.py`**: Script to evaluate the Transformer model on the test set, including generating example translations and computing the test loss.


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
- Tokenizer from scratch with EOS, BOS, UNK and PAD tokens
- Dataset preprocessing for training

## How to Use : Training

1. Ensure you have the required dependencies installed (e.g., PyTorch, NumPy).
2. Download the `fra-eng.zip` file from [ManyThings.org](http://www.manythings.org/anki/)
3. Run `train.py` to train the model:

   ```bash
   python train.py
   ```
4. To customize hyperparameters, you can specify them as command-line arguments:
   ```bash 
   python train.py --d_model 256 --num_heads 8 --dv 32 --dk 32 --d_ff 1024 --dropout 0.2 --num_encoders 6 --num_decoders 6 --batch_size 32 --learning_rate 0.0005 --num_epochs 20
   ```

Feel free to explore the code to understand the implementation and adapt it to your needs!


## Evaluation
To evaluate the model on the test set and generate example translations, run the `eval.py` script:

   ```bash
   python eval.py
   ```
The `eval.py` script will:

- Compute the average test loss.
- Generate and print example translations from the test set.

## Results
### Results for a 19M Parameter Model

The model was trained with the default hyperparameters for 10 epochs, resulting in a total of 19 million parameters. Below are the key results and observations:

- **Average Test Loss**: 1.637
- **Example Translations**:
   - **Original**: Let's all remember to be nice.
      - **Translated**: Les bactéries vivent aux oignons de suspect.
      - **Target**: Souvenons-nous tous d'être gentils.
   - **Original**: Why would I be jealous?
      - **Translated**: Pourquoi les bactéries crient.
      - **Target**: Pourquoi serais-je jaloux ?
   - **Original**: I felt like I was dead.
      - **Translated**: Je me suis senti comme les pauvres s'appauvrissent.
      - **Target**: J'avais l'impression d'être mort.
   - **Original**: The ice has melted.
      - **Translated**: Les scarabées, vivent suspect.
      - **Target**: La glace a fondu.
   - **Original**: You've got to read this.
      - **Translated**: Tu as des femmes supplémentaires qui ne va pas suspect.
      - **Target**: Il faut que tu lises ça.

### Comments on Model Performance

- The model was trained for only 10 epochs with a relatively small architecture due to computational limitations.
- The goal was not to achieve state-of-the-art (SOTA) performance but to understand the Transformer model's workings.
- Despite the limitations, the model has learned some aspects of translation, often getting the beginning of the sentence correct and sometimes the structure of the sentence.
- We used a very basic tokenizer; implementing a BPE would definitely help.
- There are still many errors, especially with longer and more complex sentences, indicating room for improvement with more training and a larger model.
### Loss Plot during Training

Below is the plot of the training and validation losses for 10 epoch for this model.

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

