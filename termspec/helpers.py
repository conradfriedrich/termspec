# -*- coding: utf-8 -*-
import string
import pickle
import pandas as pd

from nltk import FreqDist
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords as StopWords

from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np

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
    words = [w for w in words if not len(w) < 3]

    return words

def frequency_threshold(tokens, fqt = 10):
    """Return only those WORDS (i.e. unique wordforms) that appear more frequent than @fqt"""

    fq = FreqDist(tokens)
    fqt = fqt - 1
    words = list( filter( lambda x: x[1] > fqt, fq.items() ) )
    words = [item[0] for item in words]

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

def mc_indices(context_vector, fns, mc = 50):
    """Return the indices of the @mc highest values of context_vector.

    @fns is just for reference. Not really optimized.
    """
    # If the context vector has more nonzero elements than mc, only take the mc occurrences!
    if len(np.flatnonzero(context_vector)) > mc:
        fns_index_values = []
        for i, coeff in enumerate(context_vector):
            fns_index_values.append((fns[i], i, coeff))

        # Remove zero Cooccurrence Coefficient
        fns_index_values = [fiv for fiv in fns_index_values if not fiv[2] == 0]
        fns_index_values = sorted(fns_index_values, key=lambda tuple: tuple[2], reverse=True)

        indices = [fiv[1] for fiv in fns_index_values]
        indices = np.array(indices)
        indices = indices[:mc]


    else:
        indices = np.flatnonzero(context_vector)
    
    return indices




