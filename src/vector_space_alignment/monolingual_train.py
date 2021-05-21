import fasttext
import numpy as np
import pickle
from sklearn import svm
from src.vector_space_alignment.classification_methods import make_embeddings_and_target


if __name__ == '__main__':

    # choose dataset and set parameters for training
    dataset = 'toxic'
    normalize = False
    kernel = 'rbf'
    degree = 2

    ft_en = fasttext.load_model('../data/fasttext_models/wiki.en.bin')
    print('Model loaded')

    # making embeddings and their target probabilities
    X, y = make_embeddings_and_target(ft_en, dataset)

    # normalize embedding vectors to length 1 (normalizing rows - axis 1)
    if normalize:
        X = (X.T/np.linalg.norm(X, axis=1)).T

    # fitting svm
    print('Fitting SVM')
    svm_prob_predictor = svm.SVR(kernel=kernel, degree=degree)
    svm_prob_predictor.fit(X, y)

    norm_tag = 'norm_' if normalize else ''
    kernel_tag = f'{degree}' if kernel == 'poly' else kernel

    # saving model
    with open(f'../data/SVM_prob_predictor_{dataset}_{norm_tag}{kernel_tag}.pickle', 'wb') as f:
        pickle.dump(svm_prob_predictor, f)
