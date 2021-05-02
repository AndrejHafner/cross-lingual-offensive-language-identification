import numpy as np
import pandas as pd
import json
import fasttext.util


def read_fox_comments_dataset():
    """
    Read the FOX news comments dataset
    :return: Dataframe containing the comment and label for hate or not (1,0)
    """
    with open("../data/fox-news-comments.json") as f:
        df = pd.DataFrame([json.loads(line) for line in f.readlines()])

    # Remove irrelevant columns
    for col_name in ["title", "succ", "meta", "user", "mentions", "prev"]:
        del df[col_name]

    return df


def frequencies(model, data, n_hate, n_nothate):
    counts = {}
    freq = {}
    for sent, label in data.values:
        words = model.get_line(sent)[0]
        for word in words:
            word = word.lower().strip(',.?!"\'-~')
            if word == "":
                continue
            if word in counts:
                counts[word][label] += 1
            else:
                counter = [0, 0]
                counter[label] = 1
                counts[word] = counter

    for word in counts.keys():
        # normalize the change between class sizes
        counts[word][0] *= n_hate / n_nothate

        # ratio between appearances in hate speech and all appearances
        freq[word] = counts[word][1] / sum(counts[word])  # 1 if it only appears in hate, 0 if only in not_hate

    return counts, freq


if __name__ == '__main__':

    # with open('../data/hate_speech_white_supremacist_forum_dataset.json') as f:
    #     white_supr = json.load(f)

    ft_en = fasttext.load_model('../data/fasttext_models/cc.en.300.bin')

    fox = read_fox_comments_dataset()
    n_hate = fox[fox['label'] == 1].shape[0]
    n_not_hate = fox[fox['label'] == 0].shape[0]

    counts, freq = frequencies(ft_en, fox, n_hate, n_not_hate)

    stop = 0

