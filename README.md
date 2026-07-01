# Project purposes: 
For this project, we aim to train a machine model to predict the Coxeter length of a given permutation. 

There are two parts of the evaulation step. 

1. We trained the model to predict permutation with length 1 to 6 and test them on the
test data set with permutation of the same length range

2. We tested the model trained on  predicting permutation with length 1 to 6 on a new data set with permutation of length 7

# Some background information: 
Coxeter length of a permutation equals the number of inversions of a permutation 

An inversion is a pair i < j with w(i) > w(j), with w(k) being the place k gets sent to in the one-line notation for permutations.

For example, in S5, w = [3,5,1,4,2]. w(1) = 3, which means 1 gets sent to 3. 

The number of inversions for a permutation in a Symmetric group n is given by summing the elements of the inversion vector. In sage, the function is already defined. We can just use number_of_inversions(). 

link for the number_of_inversion function.

OR! use directly .length() which gives the Coxeter length. 

# Steps for this project: 
1. Generate the data set using Sage =>(generate_data.py.ipynb and data_generate_2.ipynb)
2. Decide how to represent a permutation =>(preprocess.py + data_utils.py)
3. Split the data (train, validation, and test) =>(preprocess.py)
4. Build the model =>(data_utils.py)
5. Train the model using MSE =>(train.py)
6. Evaluate using MSE =>(evaluation.py)
7. Making some plots =>(evaluation.py)
8. Reflection 