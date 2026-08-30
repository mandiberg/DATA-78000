import pandas as pd
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
import matplotlib.pyplot as plt

# testing or making model?
DO_COHERENCE_TEST = False

# model settings
NUM_TOPICS = 25
PASSES = 10
WORKERS = 8

# data settings
USE_SUBSET = True
DO_TESTING = True
TRUNCATE_SIZE = 50000
INSPECT_ROW = 4310
UNSEEN_TEXT = "The stock market is experiencing unprecedented volatility due to global economic uncertainty."

# dictonary settings
MAX_WORDS = 100000
MIN_WORDS = 15
MAX_PERCENTAGE = 0.5

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    if pd.isna(text) or not isinstance(text, str):
        return []

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def compute_coherence_values(dictionary, corpus, texts, start=2, limit=40, step=6):
    coherence_values = []
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.LdaMulticore(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=10,
            workers=1,
        )
        cm = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(cm.get_coherence())
    return coherence_values

if __name__ == '__main__':
    np.random.seed(2026)

    # load wordnet and create stemmer
    nltk.download('wordnet')
    stemmer = SnowballStemmer("english")

    # load data

    # data from: https://www.kaggle.com/datasets/therohk/million-headlines
    csv_file = 'abcnews-date-text.csv'
    headline_column = 'headline_text'

    data = pd.read_csv(csv_file)
    data_text = data[[headline_column]]

    data_text['index'] = data_text.index

    # make our dataset smaller for testing
    if USE_SUBSET:
        documents = data_text.truncate(before=1, after=TRUNCATE_SIZE)

    if DO_TESTING:
        # verify the data looks correct
        print("\n\ndocument length is ", len(documents))
        print(documents[:5])
        # look at specific entry
        doc_sample = documents[documents['index'] == INSPECT_ROW].values[0][0]
        print(f'\n\nunedited row #{INSPECT_ROW}: ')
        words = []
        for word in doc_sample.split(' '):
            words.append(word)
        print(words)
        print(f'\n\n tokenized and lemmatized row #{INSPECT_ROW}: ')
        print(preprocess(doc_sample))

    # preprocess all the docs
    processed_docs = documents[headline_column].map(preprocess)

    if DO_TESTING:
        print("\n\npreprocessed docs:")
        print(processed_docs[:10])

    # Create dictionary
    dictionary = gensim.corpora.Dictionary(processed_docs)

    if DO_TESTING:
        count = 0
        print("\n\nthe first 10 tokens in the dictionary are:")
        for k, v in dictionary.items():
            print(k, v)
            count += 1
            if count > 10:
                break

    # filter dictionary
    dictionary.filter_extremes(no_below=MIN_WORDS, no_above=MAX_PERCENTAGE, keep_n=MAX_WORDS)

    # Create Bag of Words corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if DO_TESTING:
        this_row = bow_corpus[INSPECT_ROW]
        print(f'\n\nbow corpus for row #{INSPECT_ROW}: ')
        for i in range(len(this_row)):
            print(f"Word ID: {this_row[i][0]} ({dictionary[this_row[i][0]]}) appears {this_row[i][1]} time(s).")

    # are we doing a coherence test or creating a model?
    if DO_COHERENCE_TEST:
        coherence_values = compute_coherence_values(
            dictionary=dictionary,
            corpus=bow_corpus,
            texts=processed_docs,
            start=10,
            limit=200,
            step=15
        )

        # display graph of coherence values
        x = list(range(10, 200, 15))
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.title("Topic Coherence by Number of Topics")
        plt.show()

    else:
        # create model
        try:
            print("\n\ngoing to create model")
            lda_model = gensim.models.LdaMulticore(
                bow_corpus,
                num_topics=NUM_TOPICS,
                id2word=dictionary,
                passes=PASSES,
                workers=WORKERS,
            )

            print("model created, here are the topics")
            for idx, topic in lda_model.print_topics(-1):
                print('Topic: {} \nWords: {}'.format(idx, topic))

        except Exception as e:
            print("YIKES something went wrong, quitting")
            print(str(e))
            quit()

        # use model to predict topics for unseen text
        print("\n\nusing model to predict topics for unseen text: ")
        bow_vector = dictionary.doc2bow(preprocess(UNSEEN_TEXT))
        for index, score in sorted(lda_model[bow_vector], key=lambda tup: -tup[1]):
            print(f"Score: {score}\t Topic {index}: {lda_model.print_topic(index, 5)}")



    '''
    Next Steps:
    - fine-tune the LDA model parameters
        use the coherence score to determine the best number of topics, etc.
    - load a different dataset:
        data from: https://www.kaggle.com/datasets/tmishinev/nyt-headlines-20102021
    - save lda_model to disk
    - use the model to predict topics and save them back to the dataframe/csv file
    '''