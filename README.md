# Project purposes: 
For this project, we aim to train a machine model to predict the Coxeter length of a given permutation. 

# Some background information: 
Coxeter length of a permutation equals the number of inversions of a permutation 
An inversion is a pair i < j with w(i) > w(j), with w(k) being the place k gets sent to in the one-line notation for permutations.
For example, in S5, w = [3,5,1,4,2]. w(1) = 3, which means 1 gets sent to 3. 
The number of inversions for a permutation in a Symmetric group n is given by summing the elements of the inversion vector. In sage, the function is already defined. We can just use number_of_inversions(). 
link for the number_of_inversion function.
OR! use directly .length() which gives the Coxeter length. 

# Steps for this project: 
1. Generate the data set using Sage
2. Decide how to represent a permutation 
3. Split the data (train, validation, and test)
4. Build the model 
5. Train the model using MSE
6. Evaluate using MSE
7. Making some plots
8. Reflection