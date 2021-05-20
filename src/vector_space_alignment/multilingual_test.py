import fasttext
from sklearn import svm
import pickle
import numpy as np
import pandas as pd
import json

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from src.vector_space_alignment.fasttext_word_space_alignment import load_models, make_emb_matrices
from src.vector_space_alignment.embedding_space_separation import predict_new


def load_slo_data():
    with open(f'src/data/hate_speech_slo_eval_filtered.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(columns=['id', 'type', 'content'])
    for row in data:
        df = df.append(row, ignore_index=True)

    return df


def test_on_words(ft_en, ft_slo, svm_prob_predictor, print_extremes=False, normalize=False):
    """Only used to show which slovene words have high and low hate probability when we predict from their projections
    to eng space."""
    dictionary = pd.read_csv('../data/words_dict/sl_en_test.txt', header=None, delimiter='\t').values
    dim = ft_en.get_dimension()
    X, Y = make_emb_matrices(ft_en, ft_slo, '../data/words_dict/sl_en_test.txt', dim)
    hate = ['pizda', 'kurac', 'bedak', 'idiot', 'kreten', 'terorist', 'andrej', 'čudak', 'debil', 'sovražim', 'neumno']
    for word in hate:
        X = np.hstack([X, np.reshape(ft_slo.get_word_vector(word), (300, 1))])

    # normalize embeddings
    if normalize:
        X = X / np.linalg.norm(X, axis=0)
        Y = Y / np.linalg.norm(Y, axis=0)

        with open('../data/W_norm.pickle', 'rb') as f:
            W = pickle.load(f)

    else:
        with open('../data/W_norm.pickle', 'rb') as f:
            W = pickle.load(f)

    slo_emb_to_eng = W @ X

    probabilities_slo = svm_prob_predictor.predict(slo_emb_to_eng.T)
    prob_eng = svm_prob_predictor.predict(Y.T)

    if print_extremes:
        for i in range(probabilities_slo.shape[0]):
            if probabilities_slo[i] > 0.6 or probabilities_slo[i] < 0.3:
                if i >= Y.shape[1]:
                    print(f'Word {hate[i-Y.shape[1]]} has probability: {probabilities_slo[i]}.')
                else:
                    print(f'Word {dictionary[i, 0]} has probability: {probabilities_slo[i]}. Eng: {prob_eng[i]}')

    return probabilities_slo, prob_eng


def test_on_sentences(sentences, true_type, slo_fasttext, W, svm_prob_predictor, normalize=False):

    predictions = predict_new(slo_fasttext, svm_prob_predictor, sentences, 'slo', W, normalize)

    # changing predictions to binary TODO: check thresholds
    predictions = predictions > 0.5

    print(f'Accuracy: {accuracy_score(true_type, predictions)}')
    print(f'Precision: {precision_score(true_type, predictions)}')
    print(f'Recall: {recall_score(true_type, predictions)}')
    print(f'F1 score: {f1_score(true_type, predictions)}')

    return predictions


if __name__ == '__main__':
    ft_slo = fasttext.load_model(f'../data/fasttext_models/wiki.sl.bin')

    # test_on_words(ft_en, ft_slo)

    with open('../data/W_wiki.pickle', 'rb') as f:
        W = pickle.load(f)

    with open('../data/SVM_prob_predictor_reddit.pickle', 'rb') as f:
        svm_prob_predictor = pickle.load(f)

    slo_data = load_slo_data()
    sentences = slo_data['content'].values
    true_type = slo_data['type'].values

    # changing true type to binary - all tags that are not 0 to 1
    true_type = np.minimum(true_type, 1).astype('int')

    predictions = test_on_sentences(sentences, true_type, ft_slo, W, svm_prob_predictor, True)

    t = 0

    # SVM_prob_predictor_gab, W_wiki
    # Accuracy: 0.5041443198439786
    # Precision: 0.535500157778479
    # Recall: 0.18095542759650246
    # F1 score: 0.2705029090619272

    # SVM_prob_predictor_reddit, W_wiki
    # Accuracy: 0.49195514383227695
    # Precision: 0.5
    # Recall: 0.004052036681595223
    # F1 score: 0.008038925322614765

    # SVM_prob_predictor_gab_norm_rbf, W_norm
    # Accuracy: 0.5039276233815483
    # Precision: 0.5282247765006386
    # Recall: 0.22051610151418213
    # F1 score: 0.3111412021364628

    # SVM_prob_predictor_gab_norm_2
    # predictions > 0.5)}')
    # Accuracy: 0.511024432526139
    # Precision: 0.5183639398998331
    # Recall: 0.5297504798464492
    # F1 score: 0.5239953591393314

    #  predictions > 0.45)}')
    # Accuracy: 0.5142748794625928
    # Precision: 0.5120792775888354
    # Recall: 0.9312220089571337
    # F1 score: 0.6607899515738499
