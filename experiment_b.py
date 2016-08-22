import sys
sys.path.append("termspec/")

from timer import Timer
from termspec import core as ts

import helpers as util

#####################################################################
# SETUP

read_from_file = True
filename = 'experiment_b_data.tmp'
data = None

if read_from_file:
    with Timer() as t:
        data = util.read_from_file(filename)
    print ('##### Deserialized Data in %4.1fs' % t.secs)


if not data:
    data = {}
    with Timer() as t:
        data['docs'] = ts.retrieve_data_and_tokenize(corpus = 'toy')
    print ('##### Retrieved data and Tokenized in %4.1fs' % t.secs)

    with Timer() as t:
        data['DWFM'], fns = ts.document_word_frequency_matrix(data['docs'])
    print('##### Created Document Word Frequency Matrix in %4.1fs' % t.secs)

    with Timer() as t:
        data['WWCM'], fns = ts.word_word_cooccurrence_matrix(data['docs'])
    print('##### Created Word Word Cooccurrence Matrix in %4.1fs' % t.secs)

    with Timer() as t:
        data['WWDCM'], fns = ts.word_word_dice_coeff_matrix(data['docs'])
    print('##### Created Word Word Dice Coefficient Matrix in %4.1fs' % t.secs)

    data ['fns'] = fns

    util.write_to_file(filename, data)
