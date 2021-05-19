from sklearn import svm
import pickle
import numpy as np
import pandas as pd

from fasttext_word_space_alignment import load_models, make_emb_matrices

if __name__ == '__main__':
    ft_en, ft_slo = load_models('wiki.sl.bin', 'wiki.en.bin')
    dim = ft_en.get_dimension()

    dictionary = pd.read_csv('../data/words_dict/sl_en_test.txt', header=None, delimiter='\t').values
    X, Y = make_emb_matrices(ft_en, ft_slo, '../data/words_dict/sl_en_test.txt', dim)
    hate = ['pizda', 'kurac', 'bedak', 'idiot', 'kreten', 'terorist', 'andrej', 'čudak', 'debil', 'sovražim', 'neumno']
    for word in hate:
        X = np.hstack([X, np.reshape(ft_slo.get_word_vector(word), (300, 1))])

    # normalize embeddings pravilno brez, ker pri ucenju SVM isto ne norm --> TODO: probi naucit tko da
    # X_norm = X / np.linalg.norm(X, axis=0)
    # Y_norm = Y / np.linalg.norm(Y, axis=0)

    with open('../data/W_wiki.pickle', 'rb') as f:
        W = pickle.load(f)

    with open('../data/SVM_prob_predictor_gab.pickle', 'rb') as f:
        svm_prob_predictor = pickle.load(f)

    slo_emb_to_eng = W @ X

    probabilities_slo = svm_prob_predictor.predict(slo_emb_to_eng.T)
    prob_eng = svm_prob_predictor.predict(Y.T)

    for i in range(probabilities_slo.shape[0]):
        if probabilities_slo[i] > 0.6 or probabilities_slo[i] < 0.3:
            if i >= Y.shape[1]:
                print(f'Word {hate[i-Y.shape[1]]} has probability: {probabilities_slo[i]}.')
            else:
                print(f'Word {dictionary[i, 0]} has probability: {probabilities_slo[i]}. Eng: {prob_eng[i]}')

    t = 0
