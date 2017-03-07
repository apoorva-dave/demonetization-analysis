import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as plt

from ggplot import *

pd.options.mode.chained_assignment = None


def read_data():
    # Getting data in tweets
    tweets = pd.read_csv('C:/Users/Apoorva/PycharmProjects/demonetization/input/demonetization-tweets.csv',
                         encoding="ISO-8859-1")
    # print(tweets)
    #tweets = tweets.head(20)
    return tweets


# PRE PROCESSING OF COLUMN TEXT STARTS

# we need to remove everything before :. Used re.
# m= re.search() used. ?<= ____ used for checking whether it ends with this.m.group[0] returns after pattern vala part

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

    # display(tweets['text_new'])


    # Removing urls from text
    for i in range(len(tweets['text'])):
        tweets['text_new'][i] = re.sub(r'http\S+', '', tweets['text_new'][i])

        # display(tweets['text_new'])
        # PRE PROCESSING COMPLETED


def sentiment_analysis(tweets):
    sid = SentimentIntensityAnalyzer()

    tweets['sentiment_compound_polarity'] = tweets.text_new.apply(lambda x: sid.polarity_scores(x)['compound'])
    tweets['sentiment_neutral'] = tweets.text_new.apply(lambda x: sid.polarity_scores(x)['neu'])
    tweets['sentiment_negative'] = tweets.text_new.apply(lambda x: sid.polarity_scores(x)['neg'])
    tweets['sentiment_pos'] = tweets.text_new.apply(lambda x: sid.polarity_scores(x)['pos'])
    tweets['sentiment_type'] = ''
    tweets.loc[tweets.sentiment_compound_polarity > 0, 'sentiment_type'] = 'POSITIVE'
    tweets.loc[tweets.sentiment_compound_polarity == 0, 'sentiment_type'] = 'NEUTRAL'
    tweets.loc[tweets.sentiment_compound_polarity < 0, 'sentiment_type'] = 'NEGATIVE'
    #print(tweets)
    plt.show(tweets.sentiment_type.value_counts().plot(kind='bar', title="sentiment analysis"))


def line_plot(tweets):
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    tweets['hour'] = pd.DatetimeIndex(tweets['created']).hour
    tweets['date'] = pd.DatetimeIndex(tweets['created']).date
    tweets['minute'] = pd.DatetimeIndex(tweets['created']).minute
    df = (tweets.groupby('hour', as_index=False).sentiment_compound_polarity.mean())
    print(ggplot(aes(x='hour', y='sentiment_compound_polarity'), data=df) + geom_line())


def write_to_file(tweets):
    path = 'C:/Users/Apoorva/PycharmProjects/demonetization/input/demonetization-tweets-output.csv'
    demonetization_tweets_output = open(path, 'w')
    # Writing to the file line by line:

    for i in range(len(tweets['text'])):
        demonetization_tweets_output.write(str(i) + ", ")
        demonetization_tweets_output.write(str(str(tweets['text'][i]).encode("utf-8")) + "\n")
        demonetization_tweets_output.write(str(str(tweets['favorited'][i]).encode("utf-8")) + "\n")
        demonetization_tweets_output.write(str(str(tweets['favoriteCount'][i]).encode("utf-8")) + "\n")
        demonetization_tweets_output.write(str(str(tweets['created'][i]).encode("utf-8")) + "\n")
        demonetization_tweets_output.write(str(str(tweets['truncated'][i]).encode("utf-8")) + "\n")
        demonetization_tweets_output.write(str(str(tweets['sentiment_type'][i]).encode("utf-8")) + "\n")

    # Close the file object
    demonetization_tweets_output.close()


def main():
    tweets = read_data()
    preprocessing(tweets)
    sentiment_analysis(tweets)
    line_plot(tweets)
    write_to_file(tweets)

main()
