# -*- coding: utf-8 -*-




import sys
sys.path.append("termspec/")

import helpers as util
import setup_data as sd
import scores as sc

import numpy as np
import json
from timer import Timer

score_fn = 'raw_count'
window_size = 3
filename = ''

def conduct(verbose = True, window_size = 4, corpus = 'brown', score_fn = 'dice', language = 'english'):

    ###############################################################################
    # Setup 

    anglicisms = sd.anglicisms

    data_eng = sd.easy_setup_context_window(
            fqt = 5,
            window_size = window_size,
            score_fn = score_fn,
            filename = filename,
            corpus = 'reuters',
            deserialize = True,
            serialize = True
        )
    words_eng = data_eng['words']
    # Word-Word Co-occurrence Matrix
    WWC_eng = data_eng['WWC']

    print(len(anglicisms))
    print(anglicisms)

    anglicisms = util.remove_words_not_in_list(anglicisms, words_eng, language = 'english')

    print(len(anglicisms))
    print(anglicisms)


    print('Fertig mit englisch..')
    data_ger = sd.easy_setup_context_window(
            fqt = 5,
            window_size = window_size,
            score_fn = score_fn,
            filename = filename,
            corpus = 'tiger',
            deserialize = True,
            serialize = True
        )
    words_ger = data_ger['words']
    # print(json.dumps(words_ger, indent = 4))
    # Word-Word Co-occurrence Matrix
    WWC_ger = data_ger['WWC']

    print(len(anglicisms))
    print(anglicisms)

    anglicisms = util.remove_words_not_in_list(anglicisms, words_ger, language = 'german')
    print(len(anglicisms))
    print(anglicisms)


    scores = ['nzds', 'mdcs_cosi', 'mdcs_seuc', 'sca_mdcs_cosi']
    # For each wordpair, calculate ALL the scores!


    ###############################################################################
    # Experiment
    word_scores = {}
    for word in anglicisms:
        word_eng = util.normalize([word], language = 'english')[0]
        word_ger = util.normalize([word], language = 'german')[0]
        if not word in word_scores:
            word_scores[word] = {}
            word_scores[word]['nzds'] = [
                sc.nzds(M = WWC_eng, word = word_eng, fns = words_eng),
                sc.nzds(M = WWC_ger, word = word_ger, fns = words_ger)]
            word_scores[word]['mdcs_cosi'] = [
                sc.mdcs(WWC = WWC_eng, word = word_eng, fns = words_eng, metric = 'cosine'),
                sc.mdcs(WWC = WWC_ger, word = word_ger, fns = words_ger, metric = 'cosine')]
            word_scores[word]['mdcs_seuc'] = [
                sc.mdcs(WWC = WWC_eng, word = word_eng, fns = words_eng, metric = 'seuclidean'),
                sc.mdcs(WWC = WWC_ger, word = word_ger, fns = words_ger, metric = 'seuclidean')]
            word_scores[word]['sca_mdcs_cosi'] = [
                sc.mdcs(WWC = WWC_eng, word = word_eng, fns = words_eng, metric = 'cosine', scaled = True),
                sc.mdcs(WWC = WWC_ger, word = word_ger, fns = words_ger, metric = 'cosine', scaled = True)]
        # print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))
        print(word, word_scores[word])

    ###############################################################################
    # Results

    results = np.zeros( (len(anglicisms),len(scores)), dtype=bool)

    for i, word in enumerate(anglicisms):

        for j, score in enumerate(scores):
            # Now this says TRUE or 1.0 if the english term is more general than the german
            results[i,j] = word_scores[word][score][0] > word_scores[word][score][1]


    util.printprettymatrix(M=results, rns = anglicisms, cns = scores)


    # results = np.zeros( (len(anglicisms),len(word_scores)), dtype=bool)

    # for i, pair in enumerate(word_pairs):
    #     word_a = util.normalize([pair[0]], language)[0]
    #     word_b = util.normalize([pair[1]], language)[0]
    #     for j, score in enumerate(scores):
    #         # By Convention, the More General Term comes first in word_pairs.
    #         # Checks whether the Score reflects that!
    #         results[i,j] = word_scores[word_a][score] > word_scores[word_b][score]

    # total_hits = [np.sum(results, axis = 0)[i] for i, score in enumerate(scores)]
    # percent_hits = [np.sum(results, axis = 0)[i] / len(word_pairs) for i, score in enumerate(scores)]

    # results = np.vstack([results, total_hits, percent_hits])

    # # for j, score in enumerate(scores):
    # #     results[len(word_pairs)] = np.sum(results, axis = 1)
    # labels = word_pairs + ['total hits','hit rate']


    # # Only give out detailed results in verbose mode.
    # if not verbose:
    #     results = results[-2:,:]
    #     labels = labels[-2:]

    # util.printprettymatrix(M=results, rns = labels, cns = scores)


