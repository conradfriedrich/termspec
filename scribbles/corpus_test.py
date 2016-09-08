from nltk.corpus import brown, reuters, ConllCorpusReader
import sys
sys.path.append("../termspec/")

import helpers as util

filename = 'corpustest.tmp'

tokens = util.read_from_file(filename)
if not tokens:
    print('Reading in Tiger Corpus and writing to file...')
    root = '/Users/conrad/Documents/uni/2016SS/SpinfoHausarbeit/py/termspec/corpus'
    fileid = 'tiger_aug07.conll09'
    columntypes = ['ignore', 'words', 'ignore', 'ignore', 'pos']
    tiger = ConllCorpusReader(root, fileid, columntypes, encoding='utf8')
    sents = tiger.sents()
    print(len(sents), sents[:5])
    tokens = tiger.words()
    tokens = util.normalize(tokens, language = 'german')
    util.write_to_file(filename, tokens)
    print('Done.')
else:
    print('Read Tiger Corpus from file.')

