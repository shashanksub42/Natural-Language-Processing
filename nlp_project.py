################################################################################################################################################################################

#Imports

import pandas as pd
import numpy as np
import gensim
import nltk
import pickle
import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy 
from string import punctuation
from random import shuffle
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn import metrics
from sklearn import preprocessing, svm
from sklearn import naive_bayes

################################################################################################################################################################################

#Definitions

def ingest():
    temp = []
    data = pd.read_csv('/Users/anmolukhare/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv', index_col = None, encoding = 'latin-1')
    data.columns = ['Sentiment', 'ItemID', 'TimeStamp', 'Query', 'SentimentSource', 'SentimentText']
    data.drop(['ItemID', 'TimeStamp', 'Query', 'SentimentSource'], axis = 1, inplace = True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map({4:1, 0:0})
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace = True)
    temp.insert(0, {'Sentiment':'0', 'SentimentText':'@switchfoot http://twitpic.com/2y1zl - Awww, that\'s a bummer. You shoulda got David Carr of Third Day to do it. ;D'})
    data = pd.concat([pd.DataFrame(temp), data], ignore_index = True)
    print('\nDataset loaded with shape: ', data.shape)
    return data

def WordToVec(data, n, n_dim):
    def tokenize(tweet):
        try:
            tweet = unicode(tweet.decode('utf-8').lower())
            tokens = tokenizer.tokenize(tweet)
            tokens = filter(lambda t: not t.startswith('@'), tokens)
            tokens = filter(lambda t: not t.startswith('#'), tokens)
            tokens = filter(lambda t: not t.startswith('http'), tokens)
            return tokens
        except:
            return 'NC'


    def postprocess(data, n=1600000):
        data = data.head(n)
        data['tokens'] = data['SentimentText'].apply(tokenizer.tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
        data = data[data.tokens != 'NC']
        data.reset_index(inplace=True)
        data.drop('index', inplace=True, axis=1)
        return data

    print("\nProcessing data...")
    data = postprocess(data)

    #Splitting into training and testing data
    n = 1600000
    x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens), np.array(data.head(n).Sentiment), test_size=0.2)

    def labelizeTweets(tweets, label_type):
        labelized = []
        for i,v in tqdm(enumerate(tweets)):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')


    tweet_w2v = Word2Vec(size=200, min_count=10)
    tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
    tweet_w2v.train([x.words for x in tqdm(x_train)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)
    
    temp = input("Enter a word to check most similarity matches: ")
    print("Most similar to ", temp, ": ", tweet_w2v.most_similar(temp))

    # getting a list of word vectors. limit to 10000. each is of 200 dimensions
    word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]

    # dimensionality reduction. converting the vectors to 2d vectors
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_w2v = tsne_model.fit_transform(word_vectors)

    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]
    return


def Pre_process(data):
    X = data.SentimentText
    Y = data.Sentiment
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test

def NaiveBayes(data):
    X_train, X_test, Y_train, Y_test = Pre_process(data)
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', naive_bayes.MultinomialNB())])
    print("\nFitting training data using Multinomial Naïve Bayes...")
    text_clf.fit(X_train, np.asarray(Y_train, dtype = np.float64))  
    print("\nModel fitting done.")
    pred = text_clf.predict(X_test)
    print("\nAccuracy of Multinomial Naïve Bayes model: ", np.mean(pred == Y_test)*100, "%")
    return

def SGD_Classifier(data):
    X_train, X_test, Y_train, Y_test = Pre_process(data)
    text_clf1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
    print("\nFitting training data using SGDClassifier...")
    text_clf1.fit(X_train, np.asarray(Y_train, dtype = np.float64))
    filename = 'trained_model.sav'
    pickle.dump(text_clf1, open(filename, 'wb'))
    print("\nModel fitting done.")
    pred1 = text_clf1.predict(X_test)
    print("\nAccuracy of SGD Classifier model: ", np.mean(pred1 == Y_test)*100, "%")
    return

def Grid_Search_CV(data):
    X_train, X_test, Y_train, Y_test = Pre_process(data)
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
    filename = 'trained_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    gs_clf = GridSearchCV(loaded_model, parameters, n_jobs=-1)
    print("\nFitting training data using GridSearchCV...")
    gs_clf = gs_clf.fit(X_train[:1600000], np.asarray(Y_train[:1600000], dtype = np.float64))
    print("\nModel fitting done.")
    print("\nAccuracy of GridSearchCV model: ", gs_clf.best_score_*100, "%") 
    return

################################################################################################################################################################################

#Code

print("\nLoading data...")
data = ingest()
data.drop(['index'], axis = 1, inplace = True)
n=1000000
n_dim = 200
WordToVec(data, n, n_dim)
NaiveBayes(data)
SGD_Classifier(data)
Grid_Search_CV(data)