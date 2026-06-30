# This file is basically me exploring the data set
# I made a deisgn choice to pad 0 to all the permutations to have length 7
# Because I am trying to predict the Coxeter length for permutation with length 7
# I want to fix for the permutation length for now 

import pandas as pd
import numpy as np
import ast

# load the cv
df = pd.read_csv("coxeter_length_data.csv")

#print(df.head())
#print(df.shape)

# convert permutations back to a list
df["permutation"] = df["permutation"].apply(ast.literal_eval)

#print(df["permutation"][0])
#print(type(df["permutation"][0]))

# pad zeros to all permutations to have length 7
MAX_LENGTH = 7

# Input: permutation list
# Output: permutation list with all entries having length 7
def pad_permutation(perm):
    return perm + [0] * (MAX_LENGTH - len(perm))

# create a new list
df["input"] = df["permutation"].apply(pad_permutation)

# check to see if all entries now have length 7
#print(df[["permutation", "input"]].head())

# have X being the permutations
X = df["input"].tolist()
# have y being the coxeter length 
y = df["coxeter_length"].tolist()

#print(X[0])
#print(y[0])

# since pytorch works well with numpy arrays,
# we will convert lists into arrays 

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

#print(X.shape)
#print(y.shape)

# Sanity check
#for i in range(5):
    #print(f"Original: {df['permutation'][i]}")
    #print(f"Padded : {X[i]}")
    #print(f"Length : {y[i]}")
    #print()

if __name__ == "__main__":
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")