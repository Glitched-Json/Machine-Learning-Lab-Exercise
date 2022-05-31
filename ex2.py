import nltk
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from subprocess import check_output
from sklearn.model_selection import train_test_split


class Counter:
    def __init__(self, negative=0, neutral=0, positive=0):
        self.negative = negative
        self.neutral = neutral
        self.positive = positive

    def __add__(self, b):
        return Counter(
            self.negative+b.negative,
            self.neutral+b.neutral,
            self.positive+b.positive)


nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

ps = PorterStemmer()
t = RegexpTokenizer(r'(?:\w|@)+')
data = pd.read_csv("Tweets.csv")


def tokenize(s):
    tokens = t.tokenize(s)
    tokens = [t.lower() for t in tokens]
    tokens = [ps.stem(t) for t in tokens if t not in stopwords and t[0] != '@']
    return tokens


data = data[['airline_sentiment', 'text']]
data['text'] = data['text'].map(tokenize)


train, test = train_test_split(
    data,
    test_size=0.25,
    random_state=666
)

words = dict()


def fit(row):
    tokens = row['text']
    for token in tokens:
        if token not in words:
            words[token] = Counter()
        if row['airline_sentiment'] == 'negative':
            words[token].negative += 1
        elif row['airline_sentiment'] == 'neutral':
            words[token].neutral += 1
        else:
            words[token].positive += 1


for row in train.iterrows():
    fit(row[1])


def predict(tokens):
    total = Counter()
    for token in tokens:
        if token in words:
            total += words[token]

    n = max(total.negative, total.neutral, total.positive)
    if n == total.neutral:
        return 'neutral'
    elif n == total.negative:
        return 'negative'
    else:
        return 'positive'


def accuracy(test):
    total = test.shape[0]
    count = sum(1 for _, row in test.iterrows() if row['airline_sentiment'] == predict(row['text']))
    return count/total


print(accuracy(test))


print(data)


# print(data)
