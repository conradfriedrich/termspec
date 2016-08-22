# -*- coding: utf-8 -*-

import sys
sys.path.append("termspec/")

import helpers as util

from termspec import core as ts


#####################################################################
# SETUP

filename = 'experiment_a_data.tmp'
data = ts.easy_setup(filename = filename, corpus = 'toy', deserialize = False, serialize = False)

#####################################################################
# EXPERIMENT

results = {}

sterms = []
sterms.append(util.normalize(['foo'])[0])
sterms.append(util.normalize(['plotz'])[0])

DWFM = data ['DWFM']
WWCM = data ['WWCM']
WWDICE = data ['WWDCM']

fns = data['fns']

for sterm in sterms:
    print(sterm, 'appears ', DWFM.sum(0)[fns.index(sterm)], 'time(s) in the data')

# print('Score 1: Document Frequency')
# print(sterms[0], ts.dfs(M = DWFM, word = sterms[0], fns=fns))
# print(sterms[1], ts.dfs(M = DWFM, word = sterms[1], fns=fns))
# print()

# print('Score 2: Non Zero Dimensions Score')
# print(sterms[0], ts.nzds(M = WWCM, word = sterms[0], fns = fns))
# print(sterms[1], ts.nzds(M = WWCM, word = sterms[1], fns = fns))
# print()

# print('Score 3: Simple Cooccurrence: Average Cosine Distance of Context')
# print(sterms[0], ts.tacds(WWCM = WWCM, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.tacds(WWCM = WWCM, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 4: Simple Cooccurrence: Average Cosine Distance of Context with Term')
# print(sterms[0], ts.acds(WWCM = WWCM, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.acds(WWCM = WWCM, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 5: Dice Coefficient: Average Cosine Distance of Context')
# print(sterms[0], ts.tacds(WWCM = WWDCM, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.tacds(WWCM = WWDCM, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 6: Dice Coefficient: Average Cosine Distance of Context with Term')
# print(sterms[0], ts.acds(WWCM = WWDCM, word=sterms[0], fns=fns, metric = 'cosine'))
# print(sterms[1], ts.acds(WWCM = WWDCM, word=sterms[1], fns=fns, metric = 'cosine'))
# print()

# print('Score 7: Simple Cooccurrence: Average Euclidean Distance of Context')
# print(sterms[0], ts.tacds(WWCM = WWCM, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.tacds(WWCM = WWCM, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()

# print('Score 8: Simple Cooccurrence: Average Euclidean Distance of Context with Term')
# print(sterms[0], ts.acds(WWCM = WWCM, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.acds(WWCM = WWCM, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()

# print('Score 9: Dice Coefficient: Average Euclidean Distance of Context')
# print(sterms[0], ts.tacds(WWCM = WWDCM, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.tacds(WWCM = WWDCM, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()

# print('Score 10: Dice Coefficient: Average Euclidean Distance of Context with Term')
# print(sterms[0], ts.acds(WWCM = WWDCM, word=sterms[0], fns=fns, metric = 'euclidean'))
# print(sterms[1], ts.acds(WWCM = WWDCM, word=sterms[1], fns=fns, metric = 'euclidean'))
# print()
