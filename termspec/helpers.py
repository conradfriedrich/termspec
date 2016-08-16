# -*- coding: utf-8 -*-
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords as StopWords
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

def normalize(words, language = 'english'):
	#  removes stopwords, lowercases, removes non-alphanumerics and stems (snowball)
	# words: list of strings 

	# assert not isinstance(lst, basestring)
    def ispunctuation(word):
        punctuation = string.punctuation + "„“”—–"
        for letter in word:
            if not letter in punctuation:
                return False
        return True

    stopwords = StopWords.words(language)
    stemmer = SnowballStemmer(language)

    #lowercase all terms
    words = [w.lower() for w in words]

    # stem (snowball)
    words = [stemmer.stem(w) for w in words]

    # remove all numerical terms
    words = [w for w in words if not w.isnumeric()]

    # remove pure punctuations
    words = [w for w in words if not ispunctuation(w)]

    #remove stopwords
    words = [w for w in words if not w in stopwords]

    #remove short words
    words = [w for w in words if not len(w) < 2]

    return words

def printprettymatrix(M, fns = None):

	df = pd.DataFrame(M, columns=fns, index=fns)
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)
	pd.set_option('display.expand_frame_repr', False)

	print(df)