import fasttext
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from src.vector_space_alignment.fasttext_word_space_alignment import load_models, make_emb_matrices
from src.vector_space_alignment.classification_methods import predict_new


def test_on_words(ft_en, ft_slo, svm_prob_predictor, print_extremes=False, normalize=False):
    """Only used to show which slovene words have high and low hate probability when we predict from their projections
    to eng space."""
    dictionary = pd.read_csv('../data/words_dict/sl_en_train.txt', header=None, delimiter='\t').values
    dim = ft_en.get_dimension()
    X, Y = make_emb_matrices(ft_en, ft_slo, '../data/words_dict/sl_en_train.txt', dim)

    # normalize embeddings
    if normalize:
        X = X / np.linalg.norm(X, axis=0)
        Y = Y / np.linalg.norm(Y, axis=0)

        with open('../data/W_norm.pickle', 'rb') as f:
            W = pickle.load(f)

    else:
        with open('../data/W_wiki.pickle', 'rb') as f:
            W = pickle.load(f)

    slo_emb_to_eng = W @ X

    probabilities_slo = svm_prob_predictor.predict(slo_emb_to_eng.T)
    prob_eng = svm_prob_predictor.predict(Y.T)

    if print_extremes:
        for i in range(probabilities_slo.shape[0]):
            if probabilities_slo[i] > 0.6 or probabilities_slo[i] < 0.3:
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

    dataset = 'gab'

    normalize = True if dataset == 'toxic' else False

    ft_slo = fasttext.load_model(f'../data/fasttext_models/wiki.sl.bin')

    with open('../data/W_wiki.pickle', 'rb') as f:
        W = pickle.load(f)

    with open(f'../data/SVM_prob_predictor_{dataset}.pickle', 'rb') as f:
        svm_prob_predictor = pickle.load(f)

    slo_data = pd.read_csv('../data/datasets/slo-twitter-test.csv')

    sentences = slo_data['content'].values
    true_type = slo_data['type'].values

    # changing true type to binary - all tags that are not 0 to 1
    true_type = np.minimum(true_type, 1).astype('int')

    predictions = test_on_sentences(sentences, true_type, ft_slo, W, svm_prob_predictor, normalize)
