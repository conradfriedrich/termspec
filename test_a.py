# -*- coding: utf-8 -*-
import sys
sys.path.append("termspec/")

import helpers as util
import numpy as np
import pickle

from timer import Timer

from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from termspec import core as ts

docs = ts.retrieve_data_and_tokenize(corpus = 'toy')

sents = util.flatten_documents_to_sentence_strings(docs)
count_model = CountVectorizer(ngram_range=(1,1))

# Counts each cooccurrence and returns a document-word matrix.
DWCM = count_model.fit_transform(sents)

# # These are just all the terms for later reference.
fns = count_model.get_feature_names()

util.printprettymatrix(M = DWCM.todense(), cns = fns)

# WWCM, fns = ts.word_word_cooccurrence_matrix(docs)
# util.printprettymatrix(M = WWCM, cns = fns, rns = fns)

print('############################################################################')
# DWCM2 = DWCM.copy()



# a = np.array([[ 5, 1 ,3], [ 1, 1 ,1], [ 1, 2 ,1]])
# b = np.array([1, 2, 3])

# print(a,b)
# print (a.shape, b.shape)
# print (a*b)

