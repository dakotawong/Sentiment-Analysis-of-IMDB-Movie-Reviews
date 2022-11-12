# Sentiment Analysis of IMDB Movie Reviews
This program trains a model which is used to predict the sentiment of IMDB movie reviews. The data set we have used can be found at <a href="https://ai.stanford.edu/~amaas/data/sentiment/">https://ai.stanford.edu/~amaas/data/sentiment/<a>. The model we have trained is capable of achieving an accuracy of approximately >80%.

## Extracting the Data
The .tar file has been included in the `./data` directory but can alternatively be downloaded from <a href="https://ai.stanford.edu/~amaas/data/sentiment/">here<a>. The data should be extracted from the .tar file and should result in the following directory structure:

```
├── data
│   ├── aclImdb      <==== Extracted Folder
│   │   ├── test
│   │   ├── train
│   │   ├── imdb.vocab
│   │   ├── imdbEr.txt
│   │   └── README
│   └── aclImdb_v1.tar.gz
├── models
├── utils
├── README.MD
├── test_NLP.py
└── train_NLP.py

```
Once the data has been extracted we can begin training or testing the models.

## NLTK Dependencies
It is important to note that before training or testing we also need to download NLTK modules. This can be done by running the following python file `./utils/nltk_packages.py`.

## Training Models
Models can be trained by running `train_NLP.py`. The model will then be saved in the `./models` directory as `NLP_Model.h5`.

## Testing Models
The model saved as `./models/NLP_Model.h5` can be tested by running `test_NLP.py`.