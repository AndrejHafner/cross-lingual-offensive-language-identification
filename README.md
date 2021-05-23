# Cross-lingual offensive language classification

Hateful content has become very common with the development and rapid adoption of different social platforms, hence the need for automatic detection of offensive language has grown as well. Since most of the content generated on the web is in English language, there already exist powerful systems for detection. But for languages with sparse data available, training such systems is a challenge.  Out of this came the idea of transferring the knowledge learnt on English to other languages.  In this work we show three methods for transferring the knowledge inter-language.  Models that perform well on English data show poor performance on Slovenian language, which we assume is due to the different domains of training and testing sets and to the inability of transferring the knowledge.
## Dependencies
To be able to reproduce this work, you need to install dependencies available in the `requirements.txt`.
You can create an Anaconda environment by running the following command in the root of the repository.
```
conda create --name <env_name> --file requirements.txt
```

Additionally, you will need to install the `fastText` library as described 
[here](https://fasttext.cc/docs/en/support.html) and download the pretrained
[models](https://fasttext.cc/docs/en/pretrained-vectors.html) (English and Slovene) for making word embeddings.

Filtered datasets used, and models trained with these approaches can be downloaded from 
[here](https://drive.google.com/drive/u/6/folders/1v1BVPBHT_K7bnaZN3f_W-hbcXEbZdQBS). You should save them into a folder
`data`, that should contain folders `datasets` and `words_dict`. Folder `data` should be in folder `src`. 
Datasets must be used for research purposes only and should not be further distributed.

Sources:
- [Gab & Reddit](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech) 
- [Toxic comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) 
- [TRAC-2](https://sites.google.com/view/trac2/shared-task) 
- [IMSyPP-sl](https://www.clarin.si/repository/xmlui/handle/11356/1398) (Twitter API scrapping required)

The IMSyPP-sl dataset is not added due to the Twitter's term and conditions. In order to acquire it, download the train and evaluation set 
from the provided link, create a folder `src/data/slovenian-twitter-hatespeech` and save the sets there (datasets only contain labels and Tweet IDs). Then you need to acquire a Twitter API
access key, store it into the access key variable in `src/web_scrapping/twitter_scraper.py` and run the scraper to retrieve the tweets. Next, run the script 
`src/scripts/filter_slo_tweet_dataset.py` to handle the consensus between the annotators and filter the content of tweets. For the translation 
of the dataset, run `src/multilingual_bert_classification/translation/translate_dataset.py`. In order to run Google Cloud Translation API, you first need to setup your Google Cloud account and then perform client authentication.
For all required info look [here](https://cloud.google.com/translate/docs/setup). 


## Instructions

### BERT-based approaches
To train models and test them on different datasets use the `bert_offensive_language_classification.ipynb` IPython notebook located in 
`src/multilingual_bert_classification`. Upload the notebook to Google Colab, enable GPU usage and connect the notebook to your
Google Drive containing the required models and files, which can be downloaded from [here](https://drive.google.com/drive/u/6/folders/1v1BVPBHT_K7bnaZN3f_W-hbcXEbZdQBS).
In the notebook, update the paths referring to datasets and models to the required path on your Google Drive folder (environment setup cells). You can check the structure with the
file dialog on the left.


### Vector space alignment of word embeddings and predictions based on occurrences in offensive comments
To train monolingual methods, you should select the dataset you want to train on (possible are the names of folders in 
`binary`) and run `monolingual_train.py`. To test you should do the same in `monolingual_test.py`. 
The mappings are trained by running `fasttext_word_space_alignment.py`, and you can then test the methods on Slovene data
by running `multilingual_test.py`. You can view visualizations for other datasets that are not contained in report by 
running `visualizations.py` and selecting other dataset.
If you only want to run the testing, you should save models directly into folder `data`.