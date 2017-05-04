from sklearn.grid_search import GridSearchCV

import string
import pandas as pd
from decorator import getfullargspec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import nltk
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
import random
import re
import numpy as np
import matplotlib.pyplot as plt
# Read data
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics, svm
from sklearn.metrics import roc_curve, auc
import dill
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
# from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import ShuffleSplit

pd.options.mode.chained_assignment = None

def read_data():
    data = pd.read_csv("C:/Users/Apoorva/PycharmProjects/demonetization/input/retweet_analysis.csv",  encoding="ISO-8859-1")

    print (data.shape)  # (25000, 3)
    #print (data["review"][0])  # Check out the review
    #print (data["sentiment"][0])  # Check out the sentiment (0/1)
    # data = data.head(100)
    return data

# Shuffle the data

def randomize(data):
    sentiment_data = list(zip(data["text_new"], data["isRetweet"]))
    random.shuffle(sentiment_data)
    # train_X, test_X, train_y, test_y = train_test_split(data["text"], data["isRetweet"], test_size=0.2, random_state=0)
    return sentiment_data
    # print(train_X[0])

    # return train_X,test_X,train_y,test_y
# Generate training data

def train_data(sentiment_data):
    #return random_data
    # 80% for training
    train_X, train_y = zip(*sentiment_data[:6399])
    # train_X, train_y = zip(*sentiment_data[:80])

    return list(train_X),list(train_y)

# Generate test data

def test_data(sentiment_data):
    # Keep 20% for testin
    test_X, test_y = zip(*sentiment_data[6400:8000])
    # test_X, test_y = zip(*sentiment_data[1:20])

    return list(test_X),list(test_y)

# Clean data.

def clean_text(text):
    text = text.replace("<br />", " ")
    # text = re.sub('[^A-Za-z0-9 ]+', '', str(text))
    text = text.encode().decode("utf-8")
    # Removing urls from text
    text = re.sub(r'http\S+', '', text,flags=re.MULTILINE)

    return text

def preprocessing(tweets):
    tweets['text_new'] = ''
    tweets['tweetos'] = ''

    # remove : ke pehle ka part and store in tweetos

    for i in range(len(tweets['text'])):
        try:
            tweets['tweetos'][i] = tweets['text'].str.split(':')[i][0]
            # print(tweets['tweetos'][i])
        except AttributeError:
            tweets['tweetos'][i] = 'other'

    for i in range(len(tweets['text'])):
        if tweets['tweetos'].str.contains('RT @')[i] == False:
            tweets['tweetos'][i] = 'other'

    # m.group[0] gives the text after :. stored in tweets['text_new']
    # Removing RT@

    for i in range(len(tweets['text'])):
        m = re.search('(?<=:)(.*)', tweets['text'][i])

        if tweets['text'].str.contains('RT @')[i] == True:
            try:
                tweets['text_new'][i] = m.group(0)
            except AttributeError:
                tweets['text_new'][i] = tweets['text'][i]
        else:
            tweets['text_new'][i] = tweets['text'][i]



    # Removing urls from text
    for i in range(len(tweets['text'])):
        tweets['text_new'][i] = re.sub(r'http\S+', '', tweets['text_new'][i])

        # display(tweets['text_new'])
        # PRE PROCESSING COMPLETED
    return tweets



def validation(train_X,train_y,test_X,test_y,data):
    print("validating..")

    # cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    # gammas = np.logspace(-6, -1, 10)
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier',LinearSVC())
    ])
    print("validating..")
    X = data["text_new"]
    Y = data["isRetweet"]
    scores = cross_val_score(clf, X, Y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))                #0.86+-0.4
    # estimator = svm.SVC(kernel='linear')
    # classifier = GridSearchCV(estimator,parameters)
    #
    # # classifier = GridSearchCV(clf, cv=10, param_grid=dict(gamma=gammas))
    # print(classifier)
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    #
    #
    # scores = ['precision', 'recall']
    #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
    #                        scoring='%s_macro' % score)
    #     print(clf)

        # clf.fit(X,Y)
        # print(clf.best_score_)
        # print(clf.best_score_)
    # scores = cross_val_score(classifier, X, Y, cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print(classifier.score(test_X, test_y))
    print("validating..")
    predicted = cross_val_predict(clf, test_X, test_y, cv=10)
    print(predicted)
    print(metrics.accuracy_score(test_y, predicted))            #0.84
    # clf.fit(train_X, train_y)
    # print("validating..")


def main():
    print("Reading Data..")
    data = read_data()
    print("Data Fetched")
    # print(data["review"][0])
    print("Preprocessing")
    data = preprocessing(data)
    # filename = 'preprocessed_data.sav'
    # # with open(filename, 'wb') as f:
    # #     dill.dump(data, f)
    # with open(filename, 'rb') as f:
    #     data = dill.load(f)
    print("Done")
    # train_X,test_X,train_y,test_y= randomize(data)
    sentiment_data = randomize(data)
    print("Dividing into train data..")
    train_X, train_y = train_data(sentiment_data)
    print("Dividing into test data..")
    test_X,test_y = test_data(sentiment_data)
    print("Cleaning data..")
    for i in range(0, len(train_X)):
        train_X[i] = clean_text(str(train_X[i]))
    # # print(train_X[0])
    # # print(train_y[0])
    #
    for i in range(0, len(test_X)):
        test_X[i] = clean_text(str(test_X[i]))


    validation(train_X,train_y,test_X,test_y,data)


    # unigrams(test_X,test_y,data)
    # bigrams(test_X,test_y,data)
    # build_classifiers_unigram_bigram(train_X,train_y)
    # unigram_bigram(test_X,test_y,data)



main()

