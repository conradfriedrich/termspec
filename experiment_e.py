# -*- coding: utf-8 -*-

import sys
sys.path.append("termspec/")

import helpers as util

from termspec import core as ts
import numpy as np
from timer import Timer


def raw_freq(*marginals):
    """Scores ngrams by their frequency"""
    return marginals[0]
#####################################################################
# SETUP
# categories = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

filename = 'experiment_e_data'

data = ts.easy_setup_context_window(
    fqt = 10,
    window_size = 4,
    score_fn = 'raw_count',
    filename = 'experiment_e_data',
    corpus = 'brown',
    deserialize = False,
    serialize = True
    )

words = data['words']
tokens = data['tokens']
# Word-Word Co-occurrence Matrix
WWC = data['WWC']

word_pairs = util.remove_word_pairs_not_in_corpus(ts.word_pairs, words)

scores = ['score1', 'score2']

# For each wordpair, calculate ALL the scores!
word_scores = {}
for pair in word_pairs:
    for word in pair:
        word = util.normalize([word])[0]
        if not word in word_scores:
            with Timer() as t:
                word_scores[word] = {}
                word_scores[word]['score1'] = ts.se_mdcs(WWC = WWC, word = word, fns = words)
                word_scores[word]['score2'] = ts.mdcs(WWC = WWC, word = word, fns = words, metric = 'cosine')

                # Total number of different word-Cooccurrences
            # print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))
            # print(word_scores[word])

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