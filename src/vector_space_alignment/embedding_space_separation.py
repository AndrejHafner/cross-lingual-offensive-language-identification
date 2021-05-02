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

        probs_avoid_zero_division = np.array([max(1e-5, min(1 - 1e-5, p)) for p in probs])
        logit_probs = np.log(probs_avoid_zero_division / (1 - probs_avoid_zero_division)) / logit_coef
        logit_probs = [min(100, max(-100, l)) for l in logit_probs]
        p2 = 1 / (1 + np.exp(-np.mean(logit_probs)))

        # p3 = np.prod(probs / avg_prob) / 2  # 0.91
        # p3 = np.prod(probs * 2) / 2  # multiplication by 2 so that 0.5 is neutral -- 0.94
        p3 = np.prod(probs / 0.5653) / 2  # average prob vseh besed 0.523 za fox --> 0.96

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
        probs_avoid_zero_division = np.array([max(1e-5, min(1 - 1e-5, p)) for p in probs])
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

    dataset = 'reddit'

    if dataset == 'FOX':
        data = read_fox_comments_dataset()
        data = data.rename(columns={'text': 'comment'})

    elif dataset == 'gab' or 'reddit':
        data = parse_gab_reddit_dataset(f'../data/{dataset}.csv')
    else:
        print('Unknown dataset')

    data_index = data.index.values
    np.random.seed(42)
    test_index = np.random.choice(data_index, round(0.2*len(data_index)), replace=False)

    train = data[~data.index.isin(test_index)]
    n_hate = train[train['label'] == 1].shape[0]
    n_not_hate = train[train['label'] == 0].shape[0]

    counts, probabilities = frequencies(ft_en, train, n_hate, n_not_hate)

    # comparing different options for combining word probabilities
    df, avg_prob = find_best_combination(train, ft_en, probabilities)     # --> best logit coef 4

    # making embeddings from probabilities
    X, y = make_embeddings_and_target(ft_en, probabilities)

    svm_prob_predictor = svm.SVR()
    print('Fitting SVM')
    svm_prob_predictor.fit(X, y)

    print('Testing')
    test = data[data.index.isin(test_index)]

    y_test = test['label'].values
    sentences = test['comment'].values

    y_predictions = predict_new(ft_en, svm_prob_predictor, sentences, 4)

    y_bin = y_predictions > 0.5

    print(f'Accuracy: {accuracy_score(y_test, y_bin)}')
    print(f'Precision: {precision_score(y_test, y_bin)}')
    print(f'Recall: {recall_score(y_test, y_bin)}')
    print(f'F1 score: {f1_score(y_test, y_bin)}')

    stop = 0

    # GAB data, wiki
    # Average
    # probability
    # of
    # all
    # words in train
    # set: 0.5678504185033753
    # Average: [0.56785042 0.56785042], number: [328346 328346]
    # Mean:
    # 0.881173255321606
    # 0.7951422302629046
    # Logit:
    # 0.8994187613072097
    # 0.703991685592209
    # Product:
    # 0.648138881404211
    # Product
    # inverted:
    # 0.5432849609299819
    # Fitting
    # SVM
    # Testing
    # Accuracy: 0.8489607390300231
    # Precision: 0.8834555827220864
    # Recall: 0.757247642333217
    # F1
    # score: 0.8154974609742336

    # REDDIT data, wiki
    # Average
    # probability
    # of
    # all
    # words in train
    # set: 0.5697244711031356
    # Average: [0.56972447 0.56972447], number: [212431 212431]
    # Mean:
    # 0.6136312471859523
    # 0.8861999099504727
    # Logit:
    # 0.7701485817199459
    # 0.8044236830256641
    # Product:
    # 0.8129783881134625
    # Product
    # inverted:
    # 0.7619878433138226
    # Fitting
    # SVM
    # Testing
    # Accuracy: 0.8815848716794237
    # Precision: 0.8418430884184309
    # Recall: 0.6288372093023256
    # F1
    # score: 0.7199148029818956

