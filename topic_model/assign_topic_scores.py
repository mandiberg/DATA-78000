'''
This code loads a model, opens a CSV, and assigns topic scores to the text data.
'''

import pandas as pd
import gensim


lda_model_filename = 'lda_model.gensim'
dictionary_filename = 'lda_model.dict'

csv_file = 'gc_dissertations_combined_v2.csv'
headline_column = 'abstract'


#load the model
lda_model = gensim.models.LdaMulticore.load(lda_model_filename)
dictionary = gensim.corpora.Dictionary.load(dictionary_filename)

# open the CSV
full_df = pd.read_csv(csv_file)

# drop any with missing values in the headline_column
full_df = full_df.dropna(subset=[headline_column])

# just take the first 100 rows for testing
df = full_df.head(100)

# assign topic scores to the text data in headline_column
def assign_topic_scores(text):
    if pd.isna(text) or not isinstance(text, str):
        return None, []

    # preprocess the text
    tokens = gensim.utils.simple_preprocess(text)
    bow = dictionary.doc2bow(tokens)

    # get the topic distribution for the document
    topic_distribution = lda_model.get_document_topics(bow)

    # convert to a list of topic scores
    topic_scores = [0] * lda_model.num_topics
    for topic_num, score in topic_distribution:
        topic_scores[topic_num] = score

    # return the best topic and all topic scores in a single pass
    best_topic_id, best_score = max(enumerate(topic_scores), key=lambda x: x[1], default=(None, 0.0))
    return best_topic_id, topic_scores

# apply the function once per row and split the result into separate columns
topic_results = df[headline_column].apply(assign_topic_scores)
df['topic_id'] = topic_results.apply(lambda x: x[0] if isinstance(x, tuple) else None)
df['topic_scores'] = topic_results.apply(lambda x: x[1] if isinstance(x, tuple) else [])

# print the record_id, topic_id, and topic_scores for the first few rows of the dataframe to verify
print(df[['record_id', 'topic_id', 'topic_scores']].head())