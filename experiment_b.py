# -*- coding: utf-8 -*-

import sys
sys.path.append("termspec/")

import helpers as util

from termspec import core as ts
import numpy as np
from timer import Timer

#####################################################################
# SETUP

filename = 'experiment_b_data'
data = ts.easy_setup(filename = filename, corpus = 'brown', deserialize = True, serialize = True)

word_pairs = [
    ('food','beverage'),
    ('food','dessert'),
    ('food','bread'),
    ('food','cheese'),
    ('food','meat'),
    ('food','dish'),
    ('food','butter'),
    ('food','cake'),
    ('food','egg'),
    ('food','candy'),
    ('food','pastry'),
    ('food','vegetable'),
    ('food','fruit'),

    ('vegetable', 'tomato'),
    ('vegetable', 'mushroom'),
    ('vegetable', 'legume'),

    ('vehicle','truck'),
    ('vehicle','car'),
    ('vehicle','trailer'),
    ('vehicle','campers'),

    ('person','worker'),
    ('person','writer'),
    ('person','intellectual'),
    ('person','professional'),
    ('person','leader'),
    ('person','entertainer'),
    ('person','engineer'),

    ('worker','editor'),
    ('worker','technician'),
    ('writer','journalist'),
    ('writer','commentator'),
    ('writer','novelist'),

    ('intellectual','physicist'),
    ('intellectual','historian'),
    ('intellectual','chemist'),

    ('professional','physician'),
    ('professional','educator'),
    ('professional','nurse'),
    ('professional','dentist'),

    ('entity','organism'),
    ('entity','object'),

    ('animal','dog'),
    ('animal','cat'),
    ('animal','horse'),
    ('animal','chicken'),
    ('animal','duck'),
    ('animal','fish'),
    ('animal','turtle'),
    ('animal','snake')
    ]

scores = [
    'dfs',
    'nzds',
    'sc_c_acds',
    'sc_e_acds',
    'sc_e_mdfcs',
    'dc_c_mdfcs',
    'dc_c_acds',
    'dc_e_acds',
    ]
#####################################################################
# EXPERIMENT


DWF = data ['DWF']
WWC = data ['WWC']
WWDICE = data ['WWDICE']
fns = data['fns']

# For each wordpair, calculate ALL the scores!
word_scores = {}
for pair in word_pairs:
    for word in pair:
        word = util.normalize([word])[0]
        if not word in word_scores:
            with Timer() as t:
                word_scores[word] = {}
                word_scores[word]['dfs'] = ts.dfs(M = DWF, word = word, fns=fns)
                word_scores[word]['nzds'] = ts.nzds(M = WWC, word = word, fns = fns)
                word_scores[word]['sc_c_acds'] = ts.acds(WWC = WWC, word=word, fns=fns, metric = 'cosine')
                word_scores[word]['sc_e_acds'] = ts.acds(WWC = WWC, word=word, fns=fns, metric = 'euclidean')
                word_scores[word]['sc_e_mdfcs'] = ts.mdfcs(WWC = WWC, word=word, fns=fns, metric = 'euclidean')
                word_scores[word]['dc_c_mdfcs'] = ts.mdfcs(WWC = WWDICE, word=word, fns=fns, metric = 'cosine')
                word_scores[word]['dc_c_acds'] = ts.acds(WWC = WWDICE, word=word, fns=fns, metric = 'cosine')
                word_scores[word]['dc_e_acds'] = ts.acds(WWC = WWDICE, word=word, fns=fns, metric = 'euclidean')
            print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))
print(word_scores)

results = np.zeros( (len(word_pairs),len(scores)), dtype=bool)

for i, pair in enumerate(word_pairs):
    word_a = util.normalize([pair[0]])[0]
    word_b = util.normalize([pair[1]])[0]
    for j, score in enumerate(scores):
        # By Convention, the More General Term comes first in word_pairs.
        # Checks whether the Score reflects that!
        results[i,j] = word_scores[word_a][score] > word_scores[word_b][score]

util.printprettymatrix(M=results, cns = scores, rns = word_pairs)

