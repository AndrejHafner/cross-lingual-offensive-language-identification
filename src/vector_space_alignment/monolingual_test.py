import fasttext
import pandas as pd
import pickle

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.vector_space_alignment.embedding_space_separation import predict_new

if __name__ == '__main__':
    # choose dataset and set parameters for testing
    dataset = 'fox'
    normalize = False
    kernel = 'rbf'
    degree = 2

    ft_en = fasttext.load_model('../data/fasttext_models/wiki.en.bin')
    print('Model loaded')

    test = pd.read_csv(f'../data/datasets/binary/{dataset}/test.csv')

    y_test = test['type'].values
    sentences = test['content'].values

    norm_tag = 'norm_' if normalize else ''
    kernel_tag = f'{degree}' if kernel == 'poly' else kernel

    # reading model
    with open(f'../data/SVM_prob_predictor_{dataset}_{norm_tag}{kernel_tag}.pickle', 'rb') as f:
        svm_prob_predictor = pickle.load(f)

    y_predictions = predict_new(ft_en, svm_prob_predictor, sentences, 'eng', None, normalize)

    y_bin = y_predictions > 0.5

    print(f'Accuracy: {accuracy_score(y_test, y_bin)}')
    print(f'Precision: {precision_score(y_test, y_bin)}')
    print(f'Recall: {recall_score(y_test, y_bin)}')
    print(f'F1 score: {f1_score(y_test, y_bin)}')

    stop = 0

# FOX
# rbf, norm
# Accuracy: 0.5784313725490197
# Precision: 0.36024844720496896
# Recall: 0.6904761904761905
# F1 score: 0.47346938775510206
# nenorm
# Accuracy: 0.6862745098039216
# Precision: 0.44
# Recall: 0.5238095238095238
# F1 score: 0.4782608695652174

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
#
# Fitting SVM rbf norm
# Testing
# Accuracy: 0.7832178598922248
# Precision: 0.6991513824254038
# Recall: 0.8920712539294446
# F1 score: 0.783916513198281
#
# Fitting SVM - polynomial (3)
# Testing
# Accuracy: 0.8532717474980754
# Precision: 0.9081196581196581
# Recall: 0.7422284317149843
# F1 score: 0.8168364405150874

# Fitting SVM - linear
# Testing
# Accuracy: 0.7578137028483449
# Precision: 0.9504189944134078
# Recall: 0.47537548026545584
# F1 score: 0.6337601862630967
# Fitting SVM - linear NORM
# Testing
# Accuracy: 0.789838337182448
# Precision: 0.8410746812386156
# Recall: 0.6451274886482711
# F1 score: 0.7301838307966001

# Fitting SVM - quad
# Testing
# Accuracy: 0.85635103926097
# Precision: 0.8796223446105429
# Recall: 0.7809989521480964
# F1 score: 0.8273820536540241
# Fitting SVM - 4
# Testing
# Accuracy: 0.828175519630485
# Precision: 0.9296606000983768
# Recall: 0.6601466992665037
# F1 score: 0.7720588235294119
# _____________________________
# Fitting XGB
# Testing
# Accuracy: 0.7862971516551194
# Precision: 0.8261831048208758
# Recall: 0.6524624519734544
# F1 score: 0.7291178766588603

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
#
# Fitting SVM - poly 2
# Testing
# Accuracy: 0.8624493471409275
# Precision: 0.8659305993690851
# Recall: 0.5106976744186047
# F1 score: 0.6424809830310123
# ______________________________
# Fitting XGB
# Testing
# Accuracy: 0.7793786582620441
# Precision: 0.88
# Recall: 0.10232558139534884
# F1 score: 0.18333333333333332

