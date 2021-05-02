import re

import numpy as np
import pandas as pd
import json
import fasttext.util

from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.web_scrapping.utils import remove_emojies


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
    probabilities = {}
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
        probabilities[word] = counts[word][1] / sum(counts[word])  # 1 if it only appears in hate, 0 if only in not_hate

    return counts, probabilities


def find_best_combination(train_data, model, probabilities):
    logit_coef = 4     # 0.974 na 4

    df = pd.DataFrame(columns=['label', 'p1', 'p2', 'p3', 'p4', 'sentence'])
    average = np.array([0., 0.])
    nr_words = np.array([0, 0])
    for sent, label in train_data.values:
        words = model.get_line(sent)[0]
        probs = np.array([])
        for word in words:
            word = word.lower().strip(',.?!"\'-~')
            if word == "":
                continue
            probs = np.append(probs, probabilities[word])
            average[label] += probabilities[word]
            nr_words[label] += 1

        p1 = np.mean(probs)

        probs_avoid_zero_division = np.array([min(1 - 1e-5, p) for p in probs])
        logit_probs = np.log(probs_avoid_zero_division / (1 - probs_avoid_zero_division)) / logit_coef
        logit_probs = [min(100, max(-100, l)) for l in logit_probs]
        p2 = 1 / (1 + np.exp(-np.mean(logit_probs)))

        # p3 = np.prod(probs / avg_prob) / 2  # 0.91
        # p3 = np.prod(probs * 2) / 2  # multiplication by 2 so that 0.5 is neutral -- 0.94
        p3 = np.prod(probs / 0.523) / 2  # average prob vseh besed 0.96

        # p4 = np.prod((1 - probs) / avg_prob) / 2  # probabilities turned around, so that 0 is hate speech
        p4 = np.prod((1 - probs) / 0.523) / 2

        df = df.append({'label': label, 'p1': p1, 'p2': p2, 'p3': min(p3, 1000), 'p4': p4, 'sentence': sent},
                       ignore_index=True)

    avg_prob = sum(average) / sum(nr_words)
    print(f'Average probability of all words in train set: {avg_prob}')
    average = average / nr_words
    print(f'Average: {average}, number: {nr_words}')

    print('Mean:')
    l1 = df['p1'].values > 0.5
    print(sum(l1 == df['label'].values) / df.shape[0])
    print(sum((df['p1'].values > avg_prob) == df['label'].values) / df.shape[0])

    print('Logit:')
    l2 = df['p2'].values > 0.5
    print(sum(l2 == df['label'].values) / df.shape[0])
    print(sum((df['p2'].values > avg_prob) == df['label'].values) / df.shape[0])

    print('Product:')
    l3 = df['p3'].values > 1
    print(sum(l3 == df['label'].values) / df.shape[0])

    print('Product inverted:')
    l4 = df['p1'].values < 0.2
    print(sum(l4 == df['label'].values) / df.shape[0])

    return df, avg_prob


def make_embeddings_and_target(model, probabilities):
    n = len(probabilities)
    X = np.zeros((n, model.get_dimension()))
    y = np.zeros(n)

    for i, word in enumerate(probabilities.keys()):
        X[i, :] = model.get_word_vector(word)
        y[i] = probabilities[word]

    return X, y


def predict_new(ft_model, reg_model, sentences, log_coef):
    predictions = []

    for sent in sentences:
        words = ft_model.get_line(sent)[0]
        embeddings = np.zeros((len(words), ft_model.get_dimension()))
        for i, word in enumerate(words):
            # get embedding
            word = word.lower().strip(',.?!"\'-~')
            embeddings[i, :] = ft_model.get_word_vector(word)

            # get probability embedding belongs to hate word
        probs = reg_model.predict(embeddings)

        # combine word probabilities into sentence probability
        probs_avoid_zero_division = np.array([min(1 - 1e-5, p) for p in probs])
        logit_probs = np.log(probs_avoid_zero_division / (1 - probs_avoid_zero_division)) / log_coef
        logit_probs = [min(100, max(-100, l)) for l in logit_probs]
        prob = 1 / (1 + np.exp(-np.mean(logit_probs)))

        # change into 1 and 0?
        predictions.append(prob)

    return np.array(predictions)


if __name__ == '__main__':

    # with open('../data/hate_speech_white_supremacist_forum_dataset.json') as f:
    #     white_supr = json.load(f)

    # ft_en = fasttext.load_model('../data/fasttext_models/cc.en.300.bin')
    ft_en = fasttext.load_model('../data/fasttext_models/wiki.en.bin')
    print('Model loaded')

    # need to combine more datasets
    fox = read_fox_comments_dataset()

    fox_index = fox.index.values
    np.random.seed(42)
    test_index = np.random.choice(fox_index, round(0.1*len(fox_index)), replace=False)

    fox_train = fox[~fox.index.isin(test_index)]
    n_hate = fox_train[fox_train['label'] == 1].shape[0]
    n_not_hate = fox_train[fox_train['label'] == 0].shape[0]

    counts, probabilities = frequencies(ft_en, fox_train, n_hate, n_not_hate)

    # comparing different options for combining word probabilities
    df, avg_prob = find_best_combination(fox_train, ft_en, probabilities)     # --> best logit coef 4

    # making embeddings from probabilities
    X, y = make_embeddings_and_target(ft_en, probabilities)

    svm_prob_predictor = svm.SVR()
    print('Fitting SVM')
    svm_prob_predictor.fit(X, y)

    print('Testing')
    fox_test = fox[fox.index.isin(test_index)]

    y_test = fox_test['label'].values
    sentences = fox_test['text'].values

    y_predictions = predict_new(ft_en, svm_prob_predictor, sentences, 4)

    y_bin = y_predictions > avg_prob

    print(f'Accuracy: {accuracy_score(y_test, y_bin)}')
    print(f'Precision: {precision_score(y_test, y_bin)}')
    print(f'Recall: {recall_score(y_test, y_bin)}')
    print(f'F1 score: {f1_score(y_test, y_bin)}')

    stop = 0

