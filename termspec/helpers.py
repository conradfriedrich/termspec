# -*- coding: utf-8 -*-
import string
import pickle
import pandas as pd

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords as StopWords

from nltk.tokenize import sent_tokenize, word_tokenize

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

def printprettymatrix(M, rns = None, cns = None):
    """Prints a Matrix with row and columns labels
    Matrix should be dense.

    Arguments:
    M -- Matrix to print
    rns -- Row labels
    cns -- Columnn labels

    Optional plotz says to frobnicate the bizbaz first.
    """

    df = pd.DataFrame(M, columns=cns, index=rns)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    print(df)

def flatten_documents_to_strings(docs):
    """Flattens the given documents in nested form to a string representation:

    [ #docs
        [ #document1
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ],
        [ #document2
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ]
    ]

    becomes

    [ #docs
        's1_word1 s1_word2 s1_word3 s2_word1 s2_word2 s2_word3', #document1
        's1_word1 s1_word2 s1_word3 s2_word1 s2_word2 s2_word3', #document2
    ]
    """
    strdocs = []
    for doc in docs:
        strdoc = [' '.join(sent) for sent in doc]
        strdoc = ' '.join(strdoc)
        strdocs.append(strdoc)

    return strdocs

def flatten_documents_to_sentence_strings(docs):
    """Flattens the given documents in nested form to a string representation where each sentence is a new document (useful for sentence-wise cooccurrence measuring)

    [ #docs
        [ #document1
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ],
        [ #document2
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ]
    ]

    becomes

    [ #docs
        's1_word1 s1_word2 s1_word3', #document1_sentence1
        's2_word1 s2_word2 s2_word3', #document1_sentence2
        's1_word1 s1_word2 s1_word3', #document2_sentence1
         s2_word1 s2_word2 s2_word3', #document2_sentence2
    ]
    """
    strsents = []
    for doc in docs:
        strsents.extend([' '.join(sent) for sent in doc])
    return strsents

def write_to_file(filename, data):
    """Writes the file at @filename. Does not catch errors by design, i want them kill the script."""
    filehandler = open(filename,"wb")
    pickle.dump(data,filehandler)
    filehandler.close()

def read_from_file(filename):
    """Reads the file at @filename. Does not throw FileNotFoundError """
    data = None

    try:
        file = open(filename,'rb')
        data = pickle.load(file)
        file.close()
    except FileNotFoundError as error:
        print(error)
        print('Returning empty data...')

    return data





