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


if __name__ == '__main__':
    np.random.seed(2026)

    print("load wordnet and create stemmer")
    nltk.download('wordnet')
    stemmer = SnowballStemmer("english")

    # load data

    # data from: https://www.kaggle.com/datasets/therohk/million-headlines
    csv_file = 'abcnews-date-text.csv'
    headline_column = 'headline_text'

    csv_file = 'gc_dissertations_combined_v2.csv'
    headline_column = 'abstract'

    data = pd.read_csv(csv_file)
    data_text = data[[headline_column]]

    data_text['index'] = data_text.index

    # # all the text
    # documents = data_text

    # make our dataset smaller for testing
    documents = data_text.truncate(before=1, after=50000)

    print("document length is ", len(documents))
    print(documents[:5])

    # preprocess text

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

    # look at specific entry

    doc_sample = documents[documents['index'] == 3737].values[0][0]

    print('unedited row #3737: ')
    words = []
    for word in doc_sample.split(' '):
        words.append(word)
    print(words)
    print('\n\n tokenized and lemmatized row #4310: ')
    print(preprocess(doc_sample))

    # process all the docs

    processed_docs = documents[headline_column].map(preprocess)
    processed_docs[:10]

    # Create dictionary

    dictionary = gensim.corpora.Dictionary(processed_docs)
    count = 0
    print("the first 10 tokens in the dictionary are:")
    for k, v in dictionary.items():
        print(k, v)
        count += 1
        if count > 10:
            break

    # filter dictionary

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # Create Bag of Words corpus

    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    bow_corpus[4310]

    bow_doc_4310 = bow_corpus[4310]

    print('bow corpus for row #4310: ')

    for i in range(len(bow_doc_4310)):
        print(
            "Word {} (\"{}\") appears {} time.".format(
                bow_doc_4310[i][0],
                dictionary[bow_doc_4310[i][0]],
                bow_doc_4310[i][1],
            )
        )

    # create model

    try:
        print("going to create model")

        lda_model = gensim.models.LdaMulticore(
            bow_corpus,
            num_topics=10,
            id2word=dictionary,
            passes=20,
            workers=8,
        )

        print("model created, here are the topics")

        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))

    except Exception as e:
        print("YIKES something went wrong, quitting")
        print(str(e))
        quit()


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


    coherence_values = compute_coherence_values(
        dictionary=dictionary,
        corpus=bow_corpus,
        texts=processed_docs,
        start=10,
        limit=200,
        step=15
    )

    x = list(range(10, 200, 15))
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.title("Topic Coherence by Number of Topics")
    plt.show()

    # # use CoherenceModel to evaluate the model across a range of topic sizes (e.g., $k = 5$ to $k = 50$).
    # coherence_model_lda = gensim.models.CoherenceModel(
    #     model=lda_model,
    #     texts=processed_docs,
    #     dictionary=dictionary,
    #     coherence='u_mass'
    # )

    # coherence_lda = coherence_model_lda.get_coherence()
    # print('Coherence Score: ', coherence_lda)


    # challenge 1: load a different dataset:
    # data from: https://www.kaggle.com/datasets/tmishinev/nyt-headlines-20102021

    # challenge 2: save lda_model to disk
    
    
    # challenge 3: use the model to predict topics and save them back to the dataframe/csv file