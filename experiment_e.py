# -*- coding: utf-8 -*-

import sys
sys.path.append("termspec/")

import helpers as util

from termspec import core as ts
import numpy as np
from timer import Timer
import nltk

import json

def raw_freq(*marginals):
    """Scores ngrams by their frequency"""
    return marginals[0]
#####################################################################
# SETUP
categories = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

filename = "experiment_e_data"
tokens_filename = filename + '_tokens.tmp'
words_filename = filename + '_words.tmp'

tokens = util.read_from_file(tokens_filename)
words = util.read_from_file(words_filename)

# If at least one is None...
if not (tokens and words):
    # docs = ts.retrieve_data_and_tokenize(corpus = 'brown')
    tokens = nltk.corpus.brown.words()
    tokens = util.normalize(tokens)
    tokens2 = nltk.corpus.reuters.words()
    tokens2 = util.normalize(tokens2)
    tokens.extend(tokens2)

    print('totaltokens', len(tokens))
    # docs = ts.retrieve_data_and_tokenize(corpus = 'toy', tokenize_sentences = False)

    fq = nltk.FreqDist(tokens)
    # print(fq.most_common(50))
    # print(ts.word_pairs)

    freq_threshold = 10

    freq_threshold = freq_threshold - 1
    words = list( filter( lambda x: x[1] > freq_threshold, fq.items() ) )
    words = [item[0] for item in words]

    words = set(words)
    tokens = [token for token in tokens if token in words]

    util.write_to_file(tokens_filename, tokens)
    util.write_to_file(words_filename, words)

#Flatten the document array
# tokens = [x for l in docs for x in l]

# print(docs)
print('totalwords', len(words))
print('totaltokens after substraction', len(tokens))
bgm    = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(
    tokens, window_size = 4
    )

scored = finder.score_ngrams( bgm.chi_sq )

word_pairs = []
for pair in ts.word_pairs:
    pair_in_words = True
    for word in pair:
        word = util.normalize([word])[0]
        if word not in words:
            pair_in_words = False
    if pair_in_words:
        word_pairs.append(pair)

print(len(word_pairs), len(ts.word_pairs))


M = np.zeros( (len(words), len(words)) )
print(M.shape)

# For faster Reference, Create a dictionary with words as keys and indices as values.
# Not very memory efficient, but does enhance performance.
words = list(words)
words.sort()
words_indices = {}
for i, word in enumerate(words):
    words_indices[word] = i

# Create Word-Word-Cooccurrence Matrix
for collocation in scored:
    pair = collocation[0]
    # print(pair)
    # print(words.index(pair[0]), words.index(pair[1]))
    # M[words.index(pair[0]), words.index(pair[1])] = collocation[1]
    M[words_indices[pair[0]], words_indices[pair[1]]] = collocation[1]

# print()
first = 100
util.printprettymatrix(M[:first,:first], cns = words[:first], rns = words[:first])

scores = ['score1', 'score2', ]

# For each wordpair, calculate ALL the scores!
word_scores = {}
for pair in word_pairs:
    for word in pair:
        word = util.normalize([word])[0]
        if not word in word_scores:
            with Timer() as t:
                word_scores[word] = {}
                word_scores[word]['score1'] = ts.se_mdcs(WWC = M, word = word, fns = words)
                word_scores[word]['score2'] = ts.mdcs(WWC = M, word = word, fns = words, metric = 'cosine')

                # word_scores[word]['score2'] = ts.acds(WWC = M, word = word, fns = words, metric  =  'cosine')

                # Total number of different word-Cooccurrences
            print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))
            print(word_scores[word])

#####################################################################
# RESULTS

results = np.zeros( (len(word_pairs),len(scores)), dtype=bool)

for i, pair in enumerate(word_pairs):
    word_a = util.normalize([pair[0]])[0]
    word_b = util.normalize([pair[1]])[0]
    for j, score in enumerate(scores):
        # By Convention, the More General Term comes first in word_pairs.
        # Checks whether the Score reflects that!
        results[i,j] = word_scores[word_a][score] > word_scores[word_b][score]


# for j, score in enumerate(scores):
#     results[len(word_pairs)] = np.sum(results, axis = 1)

util.printprettymatrix(M=results, rns = word_pairs, cns = scores)

print(np.sum(results, axis = 0) / results.shape[0], np.sum(results, axis = 0), results.shape[0])