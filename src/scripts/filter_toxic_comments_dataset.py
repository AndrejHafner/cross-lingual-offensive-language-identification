import re
import pandas as pd
import numpy as np

from src.utils import remove_emojies

def filter_toxic_comment_content(comment):
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'http\S+', '', comment)
    comment = re.sub(r'\t', ' ', comment)
    comment = remove_emojies(comment)
    return comment.strip()

def parse_toxic_comment_dataset(path):
    df = pd.read_csv(path)
    # appropriate --> none of the labels --> 0
    # inappropriate --> if none of the below and has toxic, severe_toxic or obscene --> 1
    # offensive --> if insult 1 or identity hate --> 2
    # violent --> if threat 1 --> 3
    data_labeled = []
    for idx, row in df.iterrows():
        type = 0
        if row["threat"] == 1:
            type = 3
        elif row["insult"] == 1 or row["identity_hate"] == 1:
            type = 2
        elif sum(row.values[2:]) > 0:
            type = 1
        data_labeled.append({"id": filter_toxic_comment_content(row["id"]), "type": type, "content": row["comment_text"]})

    df_relabeled = pd.DataFrame(data_labeled)
    print(df_relabeled["type"].value_counts())
    return df_relabeled

def balance_toxic_comment_dataset(filename, size=7500):
    df = pd.read_csv(filename)

    df_appropriate_idx = df[df["type"] == 0].index
    df_appropriate_idx_sampled = np.random.choice(df_appropriate_idx, size)

    df_selected_idx = np.hstack([np.array(df[df["type"] != 0].index), df_appropriate_idx_sampled])
    df_balanced = df.iloc[df_selected_idx, :]
    # print(df_balanced["type"].value_counts())

    return df_balanced

def train_test_split(data, test_size=0.2):
    data_index = data.index.values
    np.random.seed(42)
    test_index = np.random.choice(data_index, round(test_size * len(data_index)), replace=False)

    train = data[~data.index.isin(test_index)]
    test = data[data.index.isin(test_index)]

    return train, test

if __name__ == '__main__':
    #
    # df = parse_toxic_comment_dataset("../data/toxic-comment-classification/train.csv")
    # df.to_csv("../data/toxic-comment-classification/train_relabeled.csv", index=False, header=True)

    df_balanced = balance_toxic_comment_dataset("../data/toxic-comment-classification/train_relabeled.csv")
    train, test = train_test_split(df_balanced)

    train.to_csv("../data/toxic-comment-classification/final_balanced_multiclass/train.csv", index=False, header=True)
    test.to_csv("../data/toxic-comment-classification/final_balanced_multiclass/test.csv", index=False, header=True)

