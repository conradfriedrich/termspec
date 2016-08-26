# -*- coding: utf-8 -*-

from scipy import sparse
import sys
sys.path.append("termspec/")

import helpers as util
import math

from termspec import core as ts
import numpy as np
from timer import Timer

#####################################################################
# SETUP

filename = 'experiment_d_data.tmp'
data = ts.easy_setup(filename = filename, corpus = 'toy', deserialize = False, serialize = False)


#####################################################################
# EXPERIMENT


DWF = data ['DWF']
WWC = data ['WWC']
WWDICE = data ['WWDICE']
fns = data['fns']

# For each wordpair, calculate ALL the scores!
word_scores = {}

word_pairs = [
    # ('foo', 'bar'),
    ('foo', 'baz'),
    # ('foo', 'plotz')
    ]

scores = [
    'dfs',
    'dc_c_mdfcs_mc'
    ]

for pair in word_pairs:
    print('coc', pair[0], len(WWC[fns.index(pair[0])].nonzero()[0]))
    print('coc', pair[1], len(WWC[fns.index(pair[1])].nonzero()[0]))
    mini = min([
            len(WWC[fns.index(pair[0])].nonzero()[0]),
            len(WWC[fns.index(pair[1])].nonzero()[0])
        ])
    print('minimal occ', mini)
    for word in pair:
        word = util.normalize([word])[0]
        if not word in word_scores:
            with Timer() as t:
                word_scores[word] = {}
                word_scores[word]['dfs'] = ts.dfs(M = DWF, word = word, fns=fns)
                word_scores[word]['dc_c_mdfcs_mc'] = ts.mdfcs_mc(WWC = WWDICE, mc = mini, word = word, fns = fns, metric = 'cosine')

            print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))
            print(word_scores[word])


results = np.zeros( (len(word_pairs),len(scores)), dtype=bool)
for i, pair in enumerate(word_pairs):
    word_a = util.normalize([pair[0]])[0]
    word_b = util.normalize([pair[1]])[0]
    for j, score in enumerate(scores):
        # By Convention, the More General Term comes first in word_pairs.
        # Checks whether the Score reflects that!
        results[i,j] = word_scores[word_a][score] > word_scores[word_b][score]

print()
util.printprettymatrix(M=results, cns = scores, rns = word_pairs)

