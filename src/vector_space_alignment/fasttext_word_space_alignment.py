import fasttext.util
import pandas as pd
import numpy as np


def load_models(download=False):
    if download:
        fasttext.util.download_model('en', if_exists='ignore')  # English
        fasttext.util.download_model('sl', if_exists='ignore')  # Slovene

    ft_en = fasttext.load_model('../data/fasttext_models/cc.en.300.bin')
    ft_slo = fasttext.load_model('../data/fasttext_models/cc.sl.300.bin')

    return ft_en, ft_slo


def make_emb_matrices(model_eng, model_slo, path_to_dictionary, emb_dim):
    dictionary = pd.read_csv(path_to_dictionary, header=None, delimiter='\t').values
    n = dictionary.shape[0]
    X = np.zeros((emb_dim, n))
    Y = np.zeros((emb_dim, n))
    for i in range(n):
        ws, we = dictionary[i]
        X[:, i] = model_slo.get_word_vector(ws)
        Y[:, i] = model_eng.get_word_vector(we)
    return X, Y


if __name__ == '__main__':

    ft_en, ft_slo = load_models()
    dim = ft_en.get_dimension()
    # dim = 300
    X, Y = make_emb_matrices(ft_en, ft_slo, '../data/words_dict/sl_en_train.txt', dim)

    U, _, V = np.linalg.svd(Y @ X.T)
    W = U @ V.T


