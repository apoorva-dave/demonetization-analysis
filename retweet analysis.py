from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

import string
import pandas as pd
from decorator import getfullargspec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn import metrics, svm, tree
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
    # data = data.head(1000)
    return data

# Shuffle the data

def randomize(data):
    sentiment_data = list(zip(data["text_new"], data["isRetweet"]))
    # random.shuffle(sentiment_data)
    # train_X, test_X, train_y, test_y = train_test_split(data["text"], data["isRetweet"], test_size=0.2, random_state=0)
    return sentiment_data
    # print(train_X[0])

    # return train_X,test_X,train_y,test_y
# Generate training data

def train_data(sentiment_data):
    #return random_data
    # 80% for training
    train_X, train_y = zip(*sentiment_data[:6399])
    # train_X, train_y = zip(*sentiment_data[:800])

    return list(train_X),list(train_y)

# Generate test data

def test_data(sentiment_data):
    # Keep 20% for testin
    test_X, test_y = zip(*sentiment_data[6400:8000])
    # test_X, test_y = zip(*sentiment_data[801:1000])

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


def formatt(x):
    if x == False:
        return 0
    return 1

def validation_unigram_bigram(train_X,train_y,test_X,test_y,data):
    print("validating..")
    X = data["text_new"]
    Y = data["isRetweet"]
    # cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    # gammas = np.logspace(-6, -1, 10)
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    prediction = dict()
    clf_svm = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier',LinearSVC())
    ])
    clf_multi = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', MultinomialNB())
    ])
    clf_bernoulli = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', BernoulliNB())
    ])
    clf_logisitic = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LogisticRegression())
    ])
    clf_forest = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', RandomForestClassifier())
    ])
    #scikit-learn uses an optimised version of the CART algorithm.
    clf_cart = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', tree.DecisionTreeClassifier())
    ])
    clf_knn = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ])

    print("Predicting..")

    predicted = cross_val_predict(clf_bernoulli, test_X, test_y, cv=10)
    prediction['BernoulliNB'] = predicted

    predicted = cross_val_predict(clf_cart, test_X, test_y, cv=10)
    prediction['Cart'] = predicted

    predicted = cross_val_predict(clf_forest, test_X, test_y, cv=10)
    prediction['RandomForest'] = predicted

    predicted = cross_val_predict(clf_knn, test_X, test_y, cv=10)
    prediction['KNN'] = predicted

    predicted = cross_val_predict(clf_logisitic, test_X, test_y, cv=10)
    prediction['LogisticRegression'] = predicted
    # print(prediction['LogisticRegression'])

    predicted = cross_val_predict(clf_multi, test_X, test_y, cv=10)
    prediction['MultinomialNB'] = predicted
    # print(prediction['MultinomialNB'])

    predicted = cross_val_predict(clf_svm, test_X, test_y, cv=10)
    prediction['SVM'] = predicted
    # print(predicted)

    print("validating..")

    scores = cross_val_score(clf_bernoulli, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.80+-0.4

    scores = cross_val_score(clf_cart, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_forest, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_knn, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_logisitic, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_multi, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_svm, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    print("validating..")
    # predicted = cross_val_predict(clf, test_X, test_y, cv=10)
    # prediction['uni_svm'] = predicted
    # print(predicted)
    # print(metrics.accuracy_score(test_y, predicted))  # 0.84

    print("Confusion matrix for BernoulliNB Unigram and Bigram..")
    print(metrics.classification_report(test_y, prediction['BernoulliNB'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for Cart Unigram and Bigram..")
    print(metrics.classification_report(test_y, prediction['Cart'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for RandomForest Unigram and Bigram..")
    print(metrics.classification_report(test_y, prediction['RandomForest'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for KNN Unigram and Bigram..")
    print(metrics.classification_report(test_y, prediction['KNN'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for Logistic Regression Unigram and Bigram..")
    print(metrics.classification_report(test_y, prediction['LogisticRegression'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for MultinomialNB Unigram and Bigram..")
    print(metrics.classification_report(test_y, prediction['MultinomialNB'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for SVM Unigram and Bigram..")
    print(metrics.classification_report(test_y, prediction['SVM'], target_names=["TRUE", "FALSE"]))

    vfunc = np.vectorize(formatt)
    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k', 'darkorange', 'aqua']

    for model, predicted in prediction.items():
        # print(vfunc(predicted))
        # print(test_y)
        # print(data["Sentiment"])
        # print(np.array(data["Sentiment"]))
        # myarray = np.array(Y)
        # # print(myarray)
        newarray = []
        for i in range(0, len(test_y)):
            test_y[i] = formatt(test_y[i])
            newarray.append(test_y[i])
        # print(newarray)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(newarray, vfunc(predicted))

        # print(false_positive_rate)
        # print(true_positive_rate)




        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s:' % (model))

        cmp += 1

    plt.title('Classifiers comparaison with ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def validation_unigram(train_X, train_y, test_X, test_y, data):
    print("validating..")
    prediction = dict()
    X = data["text_new"]
    Y = data["isRetweet"]
    """
    # cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    # gammas = np.logspace(-6, -1, 10)
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    clf1 = LinearSVC()
    clf2 = MultinomialNB()
    clf3 = LogisticRegression()
    print("ewgwg")
    predicted = cross_val_predict(clf1, test_X, test_y, cv=2)
    prediction['svc'] = predicted

    predicted = cross_val_predict(clf2, test_X, test_y, cv=2)
 /   prediction['mnb'] = predicted

    predicted = cross_val_predict(clf3, test_X, test_y, cv=2)
    prediction['lr'] = predicted
    print("validating..")
    X = data["text_new"]
    Y = data["isRetweet"]
    scores = cross_val_score(clf1, X, Y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.80+-0.4
    scores = cross_val_score(clf2, X, Y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.80+-0.4
    scores = cross_val_score(clf3, X, Y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.80+-0.4

    """
    clf_svm = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LinearSVC())
    ])
    clf_multi = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', MultinomialNB())
    ])

    clf_bernoulli = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', BernoulliNB())
    ])

    clf_logisitic = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LogisticRegression())
    ])

    clf_forest = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', RandomForestClassifier())
    ])
    # scikit-learn uses an optimised version of the CART algorithm.
    clf_cart = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', tree.DecisionTreeClassifier())
    ])
    clf_knn = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ])

    print("Predicting..")

    predicted = cross_val_predict(clf_bernoulli, test_X, test_y, cv=10)
    prediction['BernoulliNB'] = predicted

    predicted = cross_val_predict(clf_cart, test_X, test_y, cv=10)
    prediction['Cart'] = predicted

    predicted = cross_val_predict(clf_forest, test_X, test_y, cv=10)
    prediction['RandomForest'] = predicted

    predicted = cross_val_predict(clf_knn, test_X, test_y, cv=10)
    prediction['KNN'] = predicted

    predicted = cross_val_predict(clf_logisitic, test_X, test_y, cv=10)
    prediction['LogisticRegression'] = predicted
    # print(prediction['LogisticRegression'])

    predicted = cross_val_predict(clf_multi, test_X, test_y, cv=10)
    prediction['MultinomialNB'] = predicted
    # print(prediction['MultinomialNB'])

    predicted = cross_val_predict(clf_svm, test_X, test_y, cv=10)
    prediction['SVM'] = predicted
    # print(predicted)

    print("validating..")


    scores = cross_val_score(clf_bernoulli, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.80+-0.4

    scores = cross_val_score(clf_cart, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_forest, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_knn, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_logisitic, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_multi, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_svm, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    print("validating..")
    # predicted = cross_val_predict(clf, test_X, test_y, cv=10)
    # prediction['uni_svm'] = predicted
    # print(predicted)
    # print(metrics.accuracy_score(test_y, predicted))  # 0.84

    print("Confusion matrix for BernoulliNB Unigram..")
    print(metrics.classification_report(test_y, prediction['BernoulliNB'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for Cart Unigram..")
    print(metrics.classification_report(test_y, prediction['Cart'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for RandomForest Unigram..")
    print(metrics.classification_report(test_y, prediction['RandomForest'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for KNN Unigram..")
    print(metrics.classification_report(test_y, prediction['KNN'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for Logistic Regression Unigram..")
    print(metrics.classification_report(test_y, prediction['LogisticRegression'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for MultinomialNB Unigram..")
    print(metrics.classification_report(test_y, prediction['MultinomialNB'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for SVM Unigram..")
    print(metrics.classification_report(test_y, prediction['SVM'], target_names=["TRUE", "FALSE"]))

    vfunc = np.vectorize(formatt)
    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k', 'darkorange', 'aqua']

    for model, predicted in prediction.items():
        # print(vfunc(predicted))
        # print(test_y)
        # print(data["Sentiment"])
        # print(np.array(data["Sentiment"]))
        # myarray = np.array(Y)
        # # print(myarray)
        newarray = []
        for i in range(0, len(test_y)):
            test_y[i] = formatt(test_y[i])
            newarray.append(test_y[i])
        # print(newarray)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(newarray, vfunc(predicted))

        # print(false_positive_rate)
        # print(true_positive_rate)
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s:' % (model))

        cmp += 1

    plt.title('Classifiers comparaison with ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



def validation_bigram(train_X, train_y, test_X, test_y, data):

    print("validating..")
    print("validating..")
    prediction = dict()
    X = data["text_new"]
    Y = data["isRetweet"]
    # cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    # gammas = np.logspace(-6, -1, 10)
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

    clf_svm = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier',LinearSVC())
    ])
    clf_multi = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', MultinomialNB())
    ])
    clf_bernoulli = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', BernoulliNB())
    ])
    clf_logisitic = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LogisticRegression())
    ])
    clf_forest = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', RandomForestClassifier())
    ])
    #scikit-learn uses an optimised version of the CART algorithm.
    clf_cart = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', tree.DecisionTreeClassifier())
    ])
    clf_knn = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ])

    print("Predicting..")


    predicted = cross_val_predict(clf_bernoulli, test_X, test_y, cv=10)
    prediction['BernoulliNB'] = predicted

    predicted = cross_val_predict(clf_cart, test_X, test_y, cv=10)
    prediction['Cart'] = predicted

    predicted = cross_val_predict(clf_forest, test_X, test_y, cv=10)
    prediction['RandomForest'] = predicted

    predicted = cross_val_predict(clf_knn, test_X, test_y, cv=10)
    prediction['KNN'] = predicted

    predicted = cross_val_predict(clf_logisitic, test_X, test_y, cv=10)
    prediction['LogisticRegression'] = predicted
    # print(prediction['LogisticRegression'])

    predicted = cross_val_predict(clf_multi, test_X, test_y, cv=10)
    prediction['MultinomialNB'] = predicted
    # print(prediction['MultinomialNB'])

    predicted = cross_val_predict(clf_svm, test_X, test_y, cv=10)
    prediction['SVM'] = predicted
    # print(predicted)

    print("validating..")

    scores = cross_val_score(clf_bernoulli, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.80+-0.4

    scores = cross_val_score(clf_cart, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_forest, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_knn, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_logisitic, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_multi, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    scores = cross_val_score(clf_svm, test_X, test_y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 0.86+-0.4

    print("validating..")
    # predicted = cross_val_predict(clf, test_X, test_y, cv=10)
    # prediction['uni_svm'] = predicted
    # print(predicted)
    # print(metrics.accuracy_score(test_y, predicted))  # 0.84

    print("Confusion matrix for BernoulliNB Bigram..")
    print(metrics.classification_report(test_y, prediction['BernoulliNB'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for Cart Bigram..")
    print(metrics.classification_report(test_y, prediction['Cart'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for RandomForest Bigram..")
    print(metrics.classification_report(test_y, prediction['RandomForest'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for KNN Bigram..")
    print(metrics.classification_report(test_y, prediction['KNN'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for Logistic Regression Bigram..")
    print(metrics.classification_report(test_y, prediction['LogisticRegression'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for MultinomialNB Bigram..")
    print(metrics.classification_report(test_y, prediction['MultinomialNB'], target_names=["TRUE", "FALSE"]))

    print("Confusion matrix for SVM Bigram..")
    print(metrics.classification_report(test_y, prediction['SVM'], target_names=["TRUE", "FALSE"]))

    vfunc = np.vectorize(formatt)
    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k', 'darkorange', 'aqua']

    for model, predicted in prediction.items():
        # print(vfunc(predicted))
        # print(test_y)
        # print(data["Sentiment"])
        # print(np.array(data["Sentiment"]))
        # myarray = np.array(Y)
        # # print(myarray)
        newarray = []
        for i in range(0, len(test_y)):
            test_y[i] = formatt(test_y[i])
            newarray.append(test_y[i])
        # print(newarray)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(newarray, vfunc(predicted))

        # print(false_positive_rate)
        # print(true_positive_rate)




        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f' % (model, roc_auc))
        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s:' % (model))

        cmp += 1

    plt.title('Classifiers comparaison with ROC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def main():
    print("Reading Data..")
    data = read_data()
    print("Data Fetched")

    # print(data["review"][0])
    print("Preprocessing")
    data = preprocessing(data)
    filename = 'preprocessed_data.sav'
    # # with open(filename, 'wb') as f:
    # #     dill.dump(data, f)
    with open(filename, 'rb') as f:
        data = dill.load(f)
    print("Done")
    # train_X,test_X,train_y,test_y= randomize(data)
    sentiment_data = randomize(data)
    print("Dividing into train data..")
    train_X, train_y = train_data(sentiment_data)
    print("Dividing into test data..")
    test_X,test_y = test_data(sentiment_data)
    print(test_y)
    # exit(0)
    print("Cleaning data..")
    for i in range(0, len(train_X)):
        train_X[i] = clean_text(str(train_X[i]))

    for i in range(0, len(test_X)):
        test_X[i] = clean_text(str(test_X[i]))

    print("For Unigram")
    validation_unigram(train_X,train_y,test_X,test_y,data)
    print("For Bigram")
    validation_bigram(train_X,train_y,test_X,test_y,data)
    print("For Unigram and Bigram")
    validation_unigram_bigram(train_X,train_y,test_X,test_y,data)

    # build_classifiers_unigram_bigram(train_X,train_y)
    # unigram_bigram(test_X,test_y,data)



main()

