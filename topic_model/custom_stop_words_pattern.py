'''
Pattern to add your own stop words
'''
# you should already have STOPWORDS imported
from gensim.parsing.preprocessing import STOPWORDS

# create a list of your own stop words
my_stop_word_list = ['mystopword1', 'mystopword2']

# create a union of the two sets (STOPWORDS is a frozen set, so you can't add to it)
# you will need to use this new set of stop words in your preprocessing function instead of the default STOPWORDS set
my_stop_words = STOPWORDS.union(set(my_stop_word_list))

# sometimes you want to get rid of A LOT of words!
# see this repo https://github.com/mandiberg/Names-Surnames-and-Countries-for-Stopwords