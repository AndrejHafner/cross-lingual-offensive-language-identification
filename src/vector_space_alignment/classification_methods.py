import pickle
import re

import numpy as np
import pandas as pd
import json
import fasttext.util


from src.utils import remove_emojies
from src.scripts.filter_toxic_comments_dataset import balance_toxic_comment_dataset


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


def filter_gab_reddit_comment(comment):
    comment = re.sub(r'http\S+', '', comment)
    comment = re.sub(r'\t', '', comment)
    comment = ".".join(comment.split(".")[1:])
    comment = remove_emojies(comment)
    return comment.strip()


def parse_gab_reddit_dataset(filename):
    df = pd.read_csv(filename, dtype=str)

    all_comments = []

    for idx, row in df.iterrows():
        try:
            comments = row["text"]
            hate_speech_idx = json.loads(row["hate_speech_idx"]) if not pd.isnull(row["hate_speech_idx"]) and "n/a" not in row["hate_speech_idx"] else []

            comments_row = [(int(el.split(".")[0]), filter_gab_reddit_comment(el)) for el in comments.split("\n") if len(el.strip()) > 0]
            comments_row_labeled = [(el[0] in hate_speech_idx, el[1]) for el in comments_row if len(el[1]) > 0]
            all_comments += comments_row_labeled
        except:
            print("Error occured, continuing..")

    df_cleaned = pd.DataFrame()
    df_cleaned["comment"] = [el[1] for el in all_comments]
    df_cleaned["label"] = [el[0] for el in all_comments]

    return df_cleaned


def split_binary_datasets():
    data_fox = read_fox_comments_dataset()
    data_fox = data_fox.rename(columns={'text': 'content', 'label': 'type'})

    data_gab = parse_gab_reddit_dataset(f'../data/gab.csv')
    data_gab = data_gab.rename(columns={'comment': 'content', 'label': 'type'})

    data_reddit = parse_gab_reddit_dataset(f'../data/reddit.csv')
    data_reddit = data_reddit.rename(columns={'comment': 'content', 'label': 'type'})

    data_toxic = balance_toxic_comment_dataset(f'../data/train_relabeled.csv', 15000)
    data_toxic = data_toxic.drop(labels='id', axis=1)
    data_toxic['type'] = np.minimum(data_toxic['type'], 1)

    for data, name in zip([data_fox, data_gab, data_reddit, data_toxic], ['fox', 'gab', 'reddit', 'toxic']):
        data_index = data.index.values
        np.random.seed(42)
        test_index = np.random.choice(data_index, round(0.2*len(data_index)), replace=False)

        train = data[~data.index.isin(test_index)]
        test = data[data.index.isin(test_index)]

        train.to_csv(f'../data/datasets/{name}/train.csv')
        test.to_csv(f'../data/datasets/{name}/test.csv')


def frequencies(model, data, train_counts=None):
    n_hate = 0
    n_nothate = 0

    if train_counts:
        # if we want to calculate true probabilities on test set, we will upgrade already known probabilities
        counts = train_counts
        test_words = {}
        for word in counts:
            n_hate += counts[word][1]
            n_nothate += counts[word][0]
    else:
        counts = {}

    probabilities = {}

    n_hate += data[data['type'] == 1].shape[0]
    n_nothate += data[data['type'] == 0].shape[0]

    for sent, label in data[['content', 'type']].values:
        words = model.get_line(sent.replace("\n", ""))[0]
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
            if train_counts:
                test_words[word] = 1

    for word in counts.keys():
        # normalize the change between class sizes
        counts[word][0] *= n_hate / n_nothate

        # ratio between appearances in hate speech and all appearances
        if train_counts:
            if word not in test_words.keys():
                # we only want to return probabilities of words in test set
                continue
        probabilities[word] = counts[word][1] / sum(counts[word])  # 1 if it only appears in hate, 0 if only in not_hate

    return counts, probabilities


def find_best_combination(train_data, model, probabilities):
    """Function used only when selecting best methods."""
    logit_coef = 4

    df = pd.DataFrame(columns=['type', 'p1', 'p2', 'p3', 'p4', 'sentence'])
    average = np.array([0., 0.])
    nr_words = np.array([0, 0])
    for sent, label in train_data[['content', 'type']].values:
        words = model.get_line(sent.replace("\n", ""))[0]
        probs = np.array([])
        for word in words:
            word = word.lower().strip(',.?!"\'-~')
            if word == "":
                continue
            probs = np.append(probs, probabilities[word])
            average[label] += probabilities[word]
            nr_words[label] += 1

        p1 = np.mean(probs)

        probs_avoid_zero_division = np.array([max(1e-5, min(1 - 1e-5, p)) for p in probs])
        logit_probs = np.log(probs_avoid_zero_division / (1 - probs_avoid_zero_division)) / logit_coef
        logit_probs = [min(100, max(-100, l)) for l in logit_probs]
        p2 = 1 / (1 + np.exp(-np.mean(logit_probs)))

        p3 = np.prod(probs / 0.5) / 2

        p4 = np.prod((1 - probs) / 0.5) / 2

        df = df.append({'type': label, 'p1': p1, 'p2': p2, 'p3': min(p3, 1000), 'p4': p4, 'sentence': sent},
                       ignore_index=True)

    avg_prob = sum(average) / sum(nr_words)
    print(f'Average probability of all words in train set: {avg_prob}')
    average = average / nr_words
    print(f'Average: {average}, number: {nr_words}')

    print('Mean:')
    l1 = df['p1'].values > 0.5
    print(sum(l1 == df['type'].values) / df.shape[0])
    print(sum((df['p1'].values > avg_prob) == df['type'].values) / df.shape[0])

    print('Logit:')
    l2 = df['p2'].values > 0.5
    print(sum(l2 == df['type'].values) / df.shape[0])
    print(sum((df['p2'].values > avg_prob) == df['type'].values) / df.shape[0])

    print('Product:')
    l3 = df['p3'].values > 1
    print(sum(l3 == df['type'].values) / df.shape[0])

    print('Product inverted:')
    l4 = df['p1'].values < 0.2
    print(sum(l4 == df['type'].values) / df.shape[0])

    return df, avg_prob


def make_embeddings_and_target(model, dataset, test=False):

    data = pd.read_csv(f'../data/datasets/binary/{dataset}/train.csv')
    data['content'] = data['content'].astype(str)
    counts, probabilities = frequencies(model, data)

    if test:
        data = pd.read_csv(f'../data/datasets/binary/{dataset}/test.csv')
        counts, probabilities = frequencies(model, data, counts)

    n = len(probabilities)
    X = np.zeros((n, model.get_dimension()))
    y = np.zeros(n)

    for i, word in enumerate(probabilities.keys()):
        X[i, :] = model.get_word_vector(word)
        y[i] = probabilities[word]

    return X, y


def sentence_probability_log(word_probabilities):
    probs_avoid_zero_division = np.array([max(1e-5, min(1 - 1e-5, p)) for p in word_probabilities])
    logit_probs = np.log(probs_avoid_zero_division / (1 - probs_avoid_zero_division))
    logit_probs = [min(100, max(-100, l)) for l in logit_probs]
    prob = 1 / (1 + np.exp(-np.mean(logit_probs)))
    return prob


def predict_new(ft_model, reg_model, sentences, lang='eng', W=None, normalize=False, sent_fun='log'):
    predictions = []

    # too speed up the predictions, we will save already calculated word probabilities
    known_prob = {}

    for sent in sentences:
        words = ft_model.get_line(sent.replace("\n", ""))[0]
        unknown_words = []
        probs = []
        for word in words:
            word = word.lower().strip(',.?!"\'-~')
            if word == '':
                continue
            if word in known_prob.keys():
                probs.append(known_prob[word])
            else:
                unknown_words.append(word)
        probs = np.array(probs)

        if len(unknown_words) > 0:
            embeddings = np.zeros((len(unknown_words), ft_model.get_dimension()))
            for i, word in enumerate(unknown_words):
                # get embedding
                embed = ft_model.get_word_vector(word)
                if normalize:
                    embed = embed / np.linalg.norm(embed)
                    # if np.isnan(sum(embed_norm)):
                    #     test = 0
                    # else:
                    #     embed = embed_norm
                embeddings[i, :] = embed

            if lang == 'slo':
                embeddings = embeddings @ W.T   # embeddings are in rows - we need to multiply from right with W.T

            emb_probs = reg_model.predict(embeddings)

            # append newly calculated probabilities to known probabilities
            for k, v in zip(unknown_words, emb_probs):
                known_prob[k] = v

            # get probability embedding belongs to hate word
            probs = np.hstack([probs, emb_probs])

        # combine word probabilities into sentence probability
        if sent_fun == 'log':
            sent_prob = sentence_probability_log(probs)
        elif sent_fun == 'mean':
            sent_prob = np.mean(probs)
        elif sent_fun == 'prod':
            sent_prob = np.prod(probs / 0.5) / 2
        else:
            print('Wrong function choice for sentence probability calculation! Choose between log, mean and prod.')
            sent_prob = 0

        predictions.append(sent_prob)

    return np.array(predictions)

