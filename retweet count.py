from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFE
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics, svm, tree, linear_model
from sklearn.metrics import roc_curve, auc
import dill
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
# from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import ShuffleSplit

pd.options.mode.chained_assignment = None

def read_data():
    data = pd.read_csv("C:/Users/Apoorva/PycharmProjects/demonetization/input/retweet_count.csv",  encoding="ISO-8859-1")

    print (data.shape)  # (25000, 3)
    # print(type(data))
    # exit(0)
    #print (data["review"][0])  # Check out the review
    #print (data["sentiment"][0])  # Check out the sentiment (0/1)
    # data = data.hea d(1000)
    return data
#
def randomize(data):
    sentiment_data = list(zip(data["text_new"], data["retweetCount"]))
    # random.shuffle(sentiment_data)
    # train_X, test_X, train_y, test_y = train_test_split(data["text"], data["isRetweet"], test_size=0.2, random_state=0)
    return sentiment_data
#     # print(train_X[0])
#
#     # return train_X,test_X,train_y,test_y
#
def train_data(sentiment_data):
    #return random_data
    # 80% for training
    train_X, train_y = zip(*sentiment_data[:6399])
    # train_X, train_y = zip(*sentiment_data[:99])
    #
    # X = sentiment_data.data[:, np.newaxis, 2]
    #
    # train_X, train_y = zip(*X[:-200])

    return list(train_X),list(train_y)
#
# # Generate test data
#
def test_data(sentiment_data):
    # Keep 20% for testin
    test_X, test_y = zip(*sentiment_data[6400:8000])
    # test_X, test_y = zip(*sentiment_data[100:199])

    return list(test_X),list(test_y)
#
#
def clean_text(text):
    text = text.replace("<br />", " ")
    # text = re.sub('[^A-Za-z0-9 ]+', '', str(text))
    text = text.encode().decode("utf-8")
    # Removing urls from text
    text = re.sub(r'http\S+', '', text,flags=re.MULTILINE)

    return text
#
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
def logistic(x, w):
  return 1.0 / (1.0 + np.exp(-x.dot(w)))
#
def count_prediction(train_X,train_y):

    regr = linear_model.LinearRegression()

    count_vect = CountVectorizer()
    train_X_counts = count_vect.fit_transform(train_X)

    # print(train_X_counts)
    tf_transformer = TfidfTransformer(use_idf=False).fit(train_X_counts)
    X_train_tf = tf_transformer.transform(train_X_counts)

    regr.fit(X_train_tf, train_y)

     # The coefficients
    print('Coefficients: \n', regr.coef_)
    print('Number of coefficients',len(regr.coef_))
#
def count_prediction_new(data):
    print("Loading")
    train_X,test_X,train_y,test_y = train_test_split(data['text_new'],data['retweetCount'],test_size = 0.33,random_state=5)
    count_prediction(train_X,train_y)
    regr = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('linear', linear_model.LinearRegression(fit_intercept=True))])

    model = regr.fit(train_X,train_y)
    print(model.predict(test_X))

    # print(model.score(test_X,test_y))           #0.979


    mean_error = np.mean((test_y - model.predict(test_X)))
    print("Mean error is ",mean_error**2)
    print('Variance score: %.2f' % model.score(test_X, test_y))
    # exit(0)

    count_vect = CountVectorizer()
    test_X_counts = count_vect.fit_transform(test_X)
    tf_transformer = TfidfTransformer(use_idf=False).fit(test_X_counts)
    X_test_tf = tf_transformer.transform(test_X_counts)

    plt.scatter(test_y,model.predict(test_X),  color='blue',s=5)

    # plt.plot(test_X, model.predict(test_X), color='black',
    #          linewidth=3)

    plt.xlabel("Retweet Counts")
    plt.ylabel("Predicted Retweet Counts")
    plt.title("Retweet Counts Vs Predicted Retweet Counts")
    plt.show()

    #
    # print(model.predict(X_test_tf))
    plt.scatter(model.predict(train_X), model.predict(train_X)-train_y, c='b',s=10,alpha=0.5)
    plt.scatter(model.predict(test_X), model.predict(test_X)-test_y, c='g',s=10)
    plt.hlines(y=0,xmin=0,xmax=1750)
    plt.ylabel("Residuals")
    plt.title("Residual plot using training(blue) and test(green) data")
    plt.show()

    plt.xticks(())
    plt.yticks(())

def main():
    print("Reading Data..")
    data = read_data()
    print("Data Fetched")

    # print(data["review"][0])
    print("Preprocessing")
    data = preprocessing(data)
    # #
    # filename = 'preprocessed_data.sav'
    # # with open(filename, 'wb') as f:
    # #     dill.dump(data, f)
    # with open(filename, 'rb') as f:
    #     data = dill.load(f)
    # print("Done")
    # train_X,test_X,train_y,test_y= randomize(data)
    # sentiment_data = randomize(data)

    # print("Dividing into train data..")
    # train_X, train_y = train_data(sentiment_data)
    # print("Dividing into test data..")
    # test_X,test_y = test_data(sentiment_data)
    #
    # # print(train_X)
    # # exit(0)
    # print("Cleaning data..")
    # for i in range(0, len(train_X)):
    #     train_X[i] = clean_text(str(train_X[i]))
    #
    # for i in range(0, len(test_X)):
    #     test_X[i] = clean_text(str(test_X[i]))

    # print(train_X.get_feature_names())
    print("Linear Regression")
    # count_prediction(train_X,train_y,test_X,test_y,data)
    count_prediction_new(data)

    # print("For Bigram")
    # validation_bigram(train_X,train_y,test_X,test_y,data)
    # print("For Unigram and Bigram")
    # validation_unigram_bigram(train_X,train_y,test_X,test_y,data)

    # build_classifiers_unigram_bigram(train_X,train_y)
    # unigram_bigram(test_X,test_y,data)

#
#
main()
