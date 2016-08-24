# -*- coding: utf-8 -*-

from scipy import sparse
import sys
sys.path.append("termspec/")

import helpers as util

from termspec import core as ts
import numpy as np
from timer import Timer

#####################################################################
# SETUP

filename = 'experiment_c_data.tmp'
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
	('foo', 'bar'),
	('foo', 'baz'),
	('foo', 'plotz')
	]

scores = [
    'dfs',
    # 'nzds',
    'dc_se_vs',
    'sc_se_vs',
    'sc_e_mdfcs'
    # 'sc_c_vs',
    # 'dc_e_vs',
    # 'dc_c_vs',
    # 'dc_c_mdfcs',
    # 'dc_c_acds',
    ]

for pair in word_pairs:
    for word in pair:
        word = util.normalize([word])[0]
        if not word in word_scores:
            with Timer() as t:
                word_scores[word] = {}
                word_scores[word]['dfs'] = ts.dfs(M = DWF, word = word, fns=fns)
                # word_scores[word]['nzds'] = ts.nzds(M = WWC, word = word, fns = fns)
                word_scores[word]['dc_se_vs'] = ts.vs(WWC = WWDICE, word=word, fns=fns)
                word_scores[word]['sc_se_vs'] = ts.vs(WWC = WWC, word=word, fns=fns)
                word_scores[word]['sc_e_mdfcs'] = ts.mdfcs(WWC = WWC, word=word, fns=fns, metric = 'euclidean')

                # word_scores[word]['dc_e_vs'] = ts.vs(WWC = WWC, word=word, fns=fns, metric = 'euclidean')
                # word_scores[word]['dc_c_vs'] = ts.vs(WWC = WWC, word=word, fns=fns, metric = 'cosine')
                # word_scores[word]['dc_c_mdfcs'] = ts.mdfcs(WWC = WWDICE, word=word, fns=fns, metric = 'cosine')
                # word_scores[word]['dc_c_acds'] = ts.acds(WWC = WWDICE, word=word, fns=fns, metric = 'cosine')
            print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))


# results = np.zeros( (len(word_pairs),len(scores) + 2), dtype=bool)
results = np.zeros( (len(word_pairs),len(scores)*2))
for i, pair in enumerate(word_pairs):
    word_a = util.normalize([pair[0]])[0]
    word_b = util.normalize([pair[1]])[0]
    j = 0
    for score in scores:
        results[i,j] = word_scores[word_a][score]
        j += 1
        results[i,j] = word_scores[word_b][score]
        j += 1

labels = []
for score in scores:
	labels.append(score + ' word_a')
	labels.append(score + ' word_b')

util.printprettymatrix(M=results, cns = labels, rns = word_pairs)