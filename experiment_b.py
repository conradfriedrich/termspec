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
    ('food','sandwich'),
    ('food','soup'),
    ('food','pizza'),
    ('food','salad'),
    ('food', 'relish'),
    ('food', 'olives'),
    ('food', 'ketchup'),
    ('food', 'cookie'),

    ('beverage', 'alcohol'),
    ('beverage', 'cola'),

    ('alcohol','liquor'),
    ('alcohol','gin'),
    ('alcohol','rum'),
    ('alcohol','brandy'),
    ('alcohol','cognac'),
    ('alcohol','wine'),
    ('alcohol','champagne'),

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
    # ('professional','educator'),
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
    'occ',
    'dfs',
    'nzds',

    # 'sc_c_mdfcs',
    # 'sc_e_mdfcs',
    # 'sc_se_mdfcs',

    # 'sc_c_mdfcs_mc',
    # 'sc_e_mdfcs_mc',
    # 'sc_se_mdfcs_mc',

    # 'sc_c_mdfcs_sca',
    # 'sc_e_mdfcs_sca',
    # 'sc_se_mdfcs_sca',

    # 'dc_c_mdfcs',
    # 'dc_e_mdfcs',
    'dc_se_mdfcs',

    # 'dc_c_mdfcs_mc',
    # 'dc_e_mdfcs_mc',
    # 'dc_se_mdfcs_mc',

    # 'dc_c_mdfcs_sca',
    # 'dc_e_mdfcs_sca',
    # 'dc_se_mdfcs_sca',
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
                # Total number of different word-Cooccurrences
                word_scores[word]['coc'] = len(WWC[fns.index(word)].nonzero()[0])

                word_scores[word]['occ'] = DWF[:, fns.index(word)].sum()
                word_scores[word]['dfs'] = ts.dfs(M = DWF, word = word, fns=fns)
                word_scores[word]['nzds'] = ts.nzds(M = WWC, word = word, fns = fns)

                # word_scores[word]['sc_c_mdfcs'] = ts.mdfcs(WWC = WWC, word = word, fns = fns, metric = 'cosine')
                # word_scores[word]['sc_e_mdfcs'] = ts.mdfcs(WWC = WWC, word = word, fns = fns, metric = 'euclidean')
                # word_scores[word]['sc_se_mdfcs'] = ts.se_mdfcs(WWC = WWC, word = word, fns = fns)

                # word_scores[word]['sc_c_mdfcs_mc'] = ts.mdfcs_mc(WWC = WWC, mc = 200, word = word, fns = fns, metric = 'cosine')
                # word_scores[word]['sc_e_mdfcs_mc'] = ts.mdfcs_mc(WWC = WWC, mc = 200, word = word, fns = fns, metric = 'euclidean')
                # word_scores[word]['sc_se_mdfcs_mc'] = ts.se_mdfcs_mc(WWC = WWC, mc = 200, word = word, fns = fns)

                # word_scores[word]['sc_c_mdfcs_sca'] = ts.mdfcs_sca(WWC = WWC, word = word, fns = fns, metric = 'cosine')
                # word_scores[word]['sc_e_mdfcs_sca'] = ts.mdfcs_sca(WWC = WWC, word = word, fns = fns, metric = 'euclidean')
                # word_scores[word]['sc_se_mdfcs_sca'] = ts.se_mdfcs_sca(WWC = WWC, word = word, fns = fns)

                # word_scores[word]['dc_c_mdfcs'] = ts.mdfcs(WWC = WWDICE, word = word, fns = fns, metric = 'cosine')
                # word_scores[word]['dc_e_mdfcs'] = ts.mdfcs(WWC = WWDICE, word = word, fns = fns, metric = 'euclidean')
                word_scores[word]['dc_se_mdfcs'] = ts.se_mdfcs(WWC = WWDICE, word = word, fns = fns)

                # word_scores[word]['dc_c_mdfcs_mc'] = ts.mdfcs_mc(WWC = WWDICE, mc = 200, word = word, fns = fns, metric = 'cosine')
                # word_scores[word]['dc_e_mdfcs_mc'] = ts.mdfcs_mc(WWC = WWDICE, mc = 200, word = word, fns = fns, metric = 'euclidean')
                # word_scores[word]['dc_se_mdfcs_mc'] = ts.se_mdfcs_mc(WWC = WWDICE, mc = 200, word = word, fns = fns)

                # word_scores[word]['dc_c_mdfcs_sca'] = ts.mdfcs_sca(WWC = WWDICE, word = word, fns = fns, metric = 'cosine')
                # word_scores[word]['dc_e_mdfcs_sca'] = ts.mdfcs_sca(WWC = WWDICE, word = word, fns = fns, metric = 'euclidean')
                # word_scores[word]['dc_se_mdfcs_sca'] = ts.se_mdfcs_sca(WWC = WWDICE, word = word, fns = fns)

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

print(np.sum(results, axis = 0) / results.shape[0])

