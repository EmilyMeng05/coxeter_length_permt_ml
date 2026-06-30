import pandas as pd
import ast

def load_data(path="coxeter_length_data.csv"):
    df = pd.read_csv(path)
    df["permutation"] = df["permutation"].apply(ast.literal_eval)
    return df

def stratified_split(df, splits=(0.7, 0.15, 0.15), seed=42):
    train_dfs, val_dfs, test_dfs = [], [], []
    for n, group in df.groupby("n"):
        group = group.sample(frac=1, random_state=seed)
        n_train = int(splits[0] * len(group))
        n_val = int(splits[1] * len(group))
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train+n_val])
        test_dfs.append(group.iloc[n_train+n_val:])
    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)

def encode_permutation(perm):
    # values are already 1..n integers; just return as a long tensor later
    return perm

if __name__ == "__main__":
    df = load_data()
    train_df, val_df, test_df = stratified_split(df)
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")