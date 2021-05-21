import fasttext
import pandas as pd
import pickle

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.vector_space_alignment.classification_methods import predict_new

if __name__ == '__main__':
    # choose dataset
    dataset = 'trac2'

    normalize = True if dataset == 'toxic' else False

    ft_en = fasttext.load_model('../data/fasttext_models/wiki.en.bin')
    print('Model loaded')

    test = pd.read_csv(f'../data/datasets/binary/{dataset}/test.csv')

    y_test = test['type'].values
    sentences = test['content'].values

    # reading model
    with open(f'../data/SVM_prob_predictor_{dataset}.pickle', 'rb') as f:
        svm_prob_predictor = pickle.load(f)

    y_predictions = predict_new(ft_en, svm_prob_predictor, sentences, 'eng', None, normalize)

    y_bin = y_predictions > 0.5

    print(f'Accuracy: {accuracy_score(y_test, y_bin)}')
    print(f'Precision: {precision_score(y_test, y_bin)}')
    print(f'Recall: {recall_score(y_test, y_bin)}')
    print(f'F1 score: {f1_score(y_test, y_bin)}')
