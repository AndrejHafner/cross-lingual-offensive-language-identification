import fasttext.util
import pandas as pd
import numpy as np
import pickle


def load_models(name_slo, name_en, download=False):
    if download:
        fasttext.util.download_model('en', if_exists='ignore')  # English
        fasttext.util.download_model('sl', if_exists='ignore')  # Slovene

    ft_en = fasttext.load_model(f'../data/fasttext_models/{name_en}')
    ft_slo = fasttext.load_model(f'../data/fasttext_models/{name_slo}')

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


def orthogonal_mapping(X, Y, normalize=False):
    if normalize:
        X = X / np.linalg.norm(X, axis=0)
        Y = Y / np.linalg.norm(Y, axis=0)

    U, _, Vt = np.linalg.svd(Y @ X.T)
    W = U @ Vt

    return W


def test_cosine_distance(X, Y, W, W_norm):
    X_test, Y_test = make_emb_matrices(ft_en, ft_slo, '../data/words_dict/sl_en_test.txt', dim)

    # normalizations
    X_norm = X / np.linalg.norm(X, axis=0)
    Y_norm = Y / np.linalg.norm(Y, axis=0)
    Xt_norm = X_test / np.linalg.norm(X_test, axis=0)
    Yt_norm = Y_test / np.linalg.norm(Y_test, axis=0)

    # calculate cosine similarity, to see if this method works and if we should normalize embeddings
    sim_not_aligned = sum(np.diag(Yt_norm.T @ Xt_norm)) / X_test.shape[1]
    sim_aligned = sum(
        np.diag((Y_test / np.linalg.norm(Y_test, axis=0)).T @ (W @ X_test / np.linalg.norm(W @ X_test, axis=0)))) / \
                  X_test.shape[1]

    sim_norm_aligned = sum(np.diag(Yt_norm.T @ (W_norm @ Xt_norm))) / X_test.shape[1]

    print(f'Average cosine similarity of Slovene and English embeddings before alignment: {sim_not_aligned}')
    print(f'Average cosine similarity of aligned embeddings: {sim_aligned}')
    print(f'Average cosine similarity of aligned embeddings with normalization: {sim_norm_aligned}')
    print('On train data:')
    print(f'Aligned train data: {sum(np.diag(Y_norm.T @ (W_norm @ X_norm))) / X_norm.shape[1]}')
    print(f'Not aligned train data: {sum(np.diag(Y_norm.T @ X_norm)) / X_norm.shape[1]}')
    print(f'Aligned train data not normalized: {sum(np.diag(Y_norm.T @ (W @ X_norm))) / X_norm.shape[1]}')


if __name__ == '__main__':

    write_mtcs = False

    ft_en, ft_slo = load_models('wiki.sl.bin', 'wiki.en.bin')

    dim = ft_en.get_dimension()

    X, Y = make_emb_matrices(ft_en, ft_slo, '../data/words_dict/sl_en_train.txt', dim)

    # linear mappings between slovene and english word space
    W = orthogonal_mapping(X, Y)
    W_norm = orthogonal_mapping(X, Y, normalize=True)

    if write_mtcs:
        with open(f'../data/W_norm.pickle', 'wb') as f:
            pickle.dump(W_norm, f)

        with open(f'../data/W_wiki.pickle', 'wb') as f:
            pickle.dump(W, f)

    test_cosine_distance(X, Y, W, W_norm)


    # WIKI
    # Average cosine similarity of Slovene and English embeddings before alignment: 0.008255468048120653
    # Average cosine similarity of aligned embeddings: 0.5114160584787307
    # Average cosine similarity of aligned embeddings with normalization: 0.5124625955303455
    # Aligned train data: 0.5695967800690408
    # Not aligned train data: 0.008180174556113636
    # Aligned train data not normalized: 0.5662968960514837

