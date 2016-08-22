import sys
sys.path.append("termspec/")

from timer import Timer
from termspec import core as ts

import helpers as util

#####################################################################
# SETUP

with Timer() as t:
    docs = ts.retrieve_data_and_tokenize(read_from_file = False)
print ('##### Retrieved data and Tokenized in %4.1fs' % t.secs)

DWFM, c_fns = ts.document_word_frequency_matrix(docs)
WWCM, c_fns = ts.word_word_cooccurrence_matrix(docs)
WWDCM, c_fns = ts.word_word_dice_coeff_matrix(docs)


# #####################################################################
# # EXPERIMENT

results = {}

sterms = []
sterms.append(util.normalize(['foo'])[0])
sterms.append(util.normalize(['plotz'])[0])


print('#####################################################################')
print('Score 1: Document Frequency')

print(sterms[0], ts.dfm(M = DWFM, word = sterms[0], fns = c_fns))
print(sterms[1], ts.dfm(M = DWFM, word = sterms[1], fns = c_fns))

print('#####################################################################')
print('Score 2: Non Zero Dimensions Score')

print(sterms[0], ts.nzdm(M = WWCM, word = sterms[0], fns = c_fns))
print(sterms[1], ts.nzdm(M = WWCM, word = sterms[1], fns = c_fns))

print('#####################################################################')
print('Score 3: Simple Cooccurrence: Average Cosine Similarity of Context')

print(sterms[0], ts.tacsm(WWCM = WWCM, word=sterms[0], fns=c_fns))
print(sterms[1], ts.tacsm(WWCM = WWCM, word=sterms[1], fns=c_fns))

print('#####################################################################')
print('Score 4: Simple Cooccurrence: Average Cosine Similarity of Context with Term')

print(sterms[0], ts.acsm(WWCM = WWCM, word=sterms[0], fns=c_fns))
print(sterms[1], ts.acsm(WWCM = WWCM, word=sterms[1], fns=c_fns))

print('#####################################################################')
print('Score 5: Simple Cooccurrence: Average Cosine Similarity of Context')

print(sterms[0], ts.tacsm(WWCM = WWDCM, word=sterms[0], fns=c_fns))
print(sterms[1], ts.tacsm(WWCM = WWDCM, word=sterms[1], fns=c_fns))

print('#####################################################################')
print('Score 6: Simple Cooccurrence: Average Cosine Similarity of Context with Term')

print(sterms[0], ts.acsm(WWCM = WWDCM, word=sterms[0], fns=c_fns))
print(sterms[1], ts.acsm(WWCM = WWDCM, word=sterms[1], fns=c_fns))