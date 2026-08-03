# Project Summary
For this current project, we are aiming to train a Transformer model that outputs the two Young tableaux (P) and (Q) produced by the Robinson–Schensted–Knuth (RSK) correspondence given a permutation.

Unlike my previous project, which predicted a single numerical value (the Coxeter length), this project is a sequence-to-sequence prediction problem. The input is a permutation, and the output is the tokenized representation of the (P) and (Q) tableaux.

The main motivation for this project is to gain hands-on experience with Transformer models while working on an algebraic combinatorics problem.

# Some goals for the project

* Get familiar with how tokenizers work.

* Design and implement my own tokenizer instead of using an existing NLP tokenizer.

* Understand how encoder-decoder Transformer architectures work.

* Learn about teacher forcing during training.

* Learn how autoregressive generation works during inference.

* Gain experience building an end-to-end deep learning project from scratch.


# Some background information

The Robinson–Schensted–Knuth (RSK) correspondence is a bijection between permutations and pairs of standard Young tableaux of the same shape.

For every permutation, the RSK insertion algorithm produces

* an insertion tableau (P),
* a recording tableau (Q).

Since Sage already implements the RSK correspondence, we use Sage to automatically generate the dataset used for training.


# How to read this repository

## 1. Data generation

See permut_rsk.ipynb

The dataset is generated using Sage.

Each row of the dataset contains

* permutation length
* permutation
* insertion tableau (P)
* recording tableau (Q)

## 2. Tokenizer

See tokenizer.py

Instead of using a natural language tokenizer, this project builds a custom tokenizer specifically for permutations and Young tableaux.

The tokenizer

* automatically builds its vocabulary from the dataset,
* converts permutations into integer tokens,
* converts the (P) and (Q) tableaux into token sequences,
* decodes predictions back into readable tableaux.

## 3. Dataset preprocessing

See preprocess.py 

This script splits the generated dataset into

* training set,
* validation set,
* testing set.

## 4. Data loading

See data_utils.py

This file

* loads the CSV files,
* tokenizes the data,
* prepares encoder inputs,
* prepares decoder inputs,
* prepares target outputs,
* pads sequences for batching.

## 5. Transformer model

See model.py

The model consists of

* token embeddings,
* positional encoding,
* Transformer encoder,
* Transformer decoder,
* linear output layer.

Teacher forcing is used during training, while autoregressive decoding is used during evaluation and inference.

## 6. Training

See train.py

The model is trained using Cross Entropy Loss.

During training we record

* training loss,
* training accuracy,
* validation accuracy.

## 7. Evaluation

See evaluation.py

The trained model is evaluated on the test dataset.

Evaluation includes

* token accuracy,
* exact sequence accuracy,
* qualitative examples comparing predicted tableaux with the ground truth.

## 8. Inference

See inference.py

This script allows a user to input a new permutation and have the trained Transformer generate the predicted (P) and (Q) tableaux.