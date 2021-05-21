import re

import numpy as np
import pandas as pd

from src.utils import remove_emojies


def filter_toxic_comment_content(comment):
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'http\S+', '', comment)
    comment = re.sub(r'\t', ' ', comment)
    comment = remove_emojies(comment)
    return comment.strip()

def train_test_split(data, test_size=0.2):
    data_index = data.index.values
    np.random.seed(42)
    test_index = np.random.choice(data_index, round(test_size * len(data_index)), replace=False)

    train = data[~data.index.isin(test_index)]
    test = data[data.index.isin(test_index)]

    return train, test
if __name__ == '__main__':
    train = pd.read_csv("../data/trac2-shared-task-dataset/eng/trac2_eng_train.csv")
    dev = pd.read_csv("../data/trac2-shared-task-dataset/eng/trac2_eng_dev.csv")
    df = pd.concat([train, dev])
    print(df["Sub-task A"].value_counts())

    type_map = {"NAG": 0, "CAG": 1, "OAG": 2}
    df_filtered = pd.DataFrame()
    df_filtered["content"] = [filter_toxic_comment_content(el) for el in df["Text"]]
    df_filtered["type"] = [type_map[el] for el in df["Sub-task A"]]

    train, test = train_test_split(df_filtered)

    train.to_csv("../data/trac2-shared-task-dataset/train.csv", index=False, header=True)
    test.to_csv("../data/trac2-shared-task-dataset/test.csv", index=False, header=True)

