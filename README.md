# Offensive-language-exploratory-analysis

Today automatic hate speech detection is an important problem on online platforms. There is a lot of English data to 
train models on, but for some languages we can't afford to train expensive models to be able to detect offensive 
language. In this work we show a couple of methods that can combine training on English data with transfer to Slovene. 
With these models we achieve good performance on English data, but are not able to get good results on Slovene.

## Dependencies
To be able to reproduce this work, you need to install dependencies available in the `requirements.txt`.
Additionally, you will need to install the `fastText` library as described 
[here](https://fasttext.cc/docs/en/support.html) and download the pretrained
[models](https://fasttext.cc/docs/en/pretrained-vectors.html) (English and Slovene) for making word embeddings.

Datasets used, and models trained with these approaches can be downloaded from 
[here](https://drive.google.com/drive/u/6/folders/1v1BVPBHT_K7bnaZN3f_W-hbcXEbZdQBS). You should save them into a folder
`data`, that should contain folders `datasets` and `words_dict`. Folder `data` should be in folder `src`.

## Instructions

### Andrejevi BERTi

### Vector space alignment of word embeddings and predictions based on occurrences in offensive comments
To train monolingual methods, you should select the dataset you want to train on (possible are the names of folders in 
`binary`) and run `monolingual_train.py`. To test you should do the same in `monolingual_test.py`. 
The mappings are trained by running `fasttext_word_space_alignment.py`, and you can then test the methods on Slovene data
by running `multilingual_test.py`. You can view visualizations for other datasets that are not contained in report by 
running `visualizations.py` and selecting other dataset.
If you only want to run the testing, you should save models directly into folder `data`.