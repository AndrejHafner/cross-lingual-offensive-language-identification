from classification_methods import make_embeddings_and_target
from multilingual_test import test_on_words

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pickle


def visualize_mapping(ft_slo, ft_eng, svm_predictor):
    prob_slo, prob_eng = test_on_words(ft_slo, ft_eng, svm_predictor, normalize=True)

    plt.scatter(prob_slo, prob_eng, s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('SVM prediction of W*X')
    plt.ylabel('SVM prediction of Y')
    plt.show()


def visualize_svm(ft_en, svm_pred, dataset, normalize=False):
    # making embeddings and calculating true labels for words from test set
    X, y = make_embeddings_and_target(ft_en, dataset, test=True)

    if normalize:
        X = (X.T / np.linalg.norm(X, axis=1)).T
    # predicting labels with SVM
    y_pred = svm_pred.predict(X)

    plt.figure(figsize=(5, 6))
    plt.scatter(y, y_pred, s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylim([-0.2, 1.2])
    plt.xlabel('P(w is offensive)')
    plt.ylabel('SVM prediction of P(w is offensive)')
    plt.show()


if __name__ == '__main__':
    ft_slo = fasttext.load_model(f'../data/fasttext_models/wiki.sl.bin')
    ft_eng = fasttext.load_model(f'../data/fasttext_models/wiki.en.bin')

    dataset = 'toxic'
    normalize = True if dataset == 'toxic' else False

    with open(f'../data/SVM_prob_predictor_{dataset}.pickle', 'rb') as f:
        svm_prob_predictor = pickle.load(f)

    visualize_mapping(ft_slo, ft_eng, svm_prob_predictor)

    visualize_svm(ft_eng, svm_prob_predictor, dataset, normalize=normalize)

