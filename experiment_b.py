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

filename = 'experiment_b_data.tmp'
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
    # 'sc_c_tacds', 
    'sc_c_acds',
    # 'sc_e_tacds', 
    'sc_e_acds',
    'dc_c_tacds', 
    'dc_c_acds',
    # 'dc_e_tacds', 
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
                word_scores[word]['dc_c_tacds'] = ts.tacds(WWC = WWDICE, word=word, fns=fns, metric = 'cosine')
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
# print()
# for sterm in sterms:
#     print(sterm, 'appears ', DWF.sum(0)[fns.index(sterm)], 'time(s) in the data')

# print()
# print('Each Score should show: lower (absolute) Score = higher Specificity')
# print()
# print('Score 1: Document Frequency Score')
# print(sterms[0], ts.dfs(M = DWF, word = sterms[0], fns=fns))
# print(sterms[1], ts.dfs(M = DWF, word = sterms[1], fns=fns))
# print()

# print('Score 2: Non Zero Dimensions Score')
# print(sterms[0], ts.nzds(M = WWC, word = sterms[0], fns = fns))
# print(sterms[1], ts.nzds(M = WWC, word = sterms[1], fns = fns))
# print()

# print('Score 3: Simple Cooccurrence: Average Cosine Distance of Context')
# print(sterms[0], ts.tacds(WWC = WWC, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.tacds(WWC = WWC, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 4: Simple Cooccurrence: Average Cosine Distance of Context with Term')
# print(sterms[0], ts.acds(WWC = WWC, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.acds(WWC = WWC, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 5: Dice Coefficient: Average Cosine Distance of Context')
# print(sterms[0], ts.tacds(WWC = WWDICE, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.tacds(WWC = WWDICE, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 6: Dice Coefficient: Average Cosine Distance of Context with Term')
# print(sterms[0], ts.acds(WWC = WWDICE, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.acds(WWC = WWDICE, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 7: Simple Cooccurrence: Average Euclidean Distance of Context')
# print(sterms[0], ts.tacds(WWC = WWC, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.tacds(WWC = WWC, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()

# print('Score 8: Simple Cooccurrence: Average Euclidean Distance of Context with Term')
# print(sterms[0], ts.acds(WWC = WWC, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.acds(WWC = WWC, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()

# print('Score 9: Dice Coefficient: Average Euclidean Distance of Context')
# print(sterms[0], ts.tacds(WWC = WWDICE, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.tacds(WWC = WWDICE, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()

# print('Score 10: Dice Coefficient: Average Euclidean Distance of Context with Term')
# print(sterms[0], ts.acds(WWC = WWDICE, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.acds(WWC = WWDICE, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()
