from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from embedding_space_separation import make_embeddings_and_target, split_binary_datasets
from multilingual_test import test_on_words

import fasttext
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pickle


def visualize_eng_word_space(model, dataset):
    X, y, _ = make_embeddings_and_target(model, dataset)

    sample = np.random.randint(X.shape[0], size=10000)

    pca = PCA(50)
    X_small = pca.fit_transform(X[sample, :])

    tsne = TSNE(3, n_jobs=2)
    X_3d = tsne.fit_transform(X_small)

    df = pd.DataFrame(columns=['tsne1', 'tsne2', 'tsne3', 'prob'])
    df['tsne1'] = X_3d[:, 0]
    df['tsne2'] = X_3d[:, 1]
    df['tsne3'] = X_3d[:, 2]
    df['prob'] = y[sample]

    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(data=df, x="tsne1", y="tsne2", hue="prob", palette=cmap, alpha=0.3)
    plt.show()

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["tsne1"],
        ys=df["tsne2"],
        zs=df["tsne3"],
        c=df["prob"],
        cmap=cmap, alpha=0.3
    )
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    plt.show()

    df2 = pd.DataFrame(columns=['pca1', 'pca2', 'pca3', 'prob'])
    df2['pca1'] = X_small[:, 0]
    df2['pca2'] = X_small[:, 1]
    df2['pca3'] = X_small[:, 2]
    df2['prob'] = y[sample]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(data=df2, x="pca1", y="pca2", hue="prob", palette=cmap, alpha=0.3)
    plt.show()

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df2["pca1"],
        ys=df2["pca2"],
        zs=df2["pca3"],
        c=df2["prob"],
        cmap=cmap, alpha=0.3
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()


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
    # visualize_eng_word_space(ft_eng, 'gab')

    with open('../data/SVM_prob_predictor_toxic_norm_rbf.pickle', 'rb') as f:
        svm_prob_predictor = pickle.load(f)

    visualize_mapping(ft_slo, ft_eng, svm_prob_predictor)
    # split_binary_datasets()
    visualize_svm(ft_eng, svm_prob_predictor, 'toxic', normalize=True)

