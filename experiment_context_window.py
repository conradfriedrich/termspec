# -*- coding: utf-8 -*-

import sys
sys.path.append("termspec/")

import helpers as util
import setup_data as sd
import scores as sc

import numpy as np
from timer import Timer


def conduct(verbose = True, window_size = 4, corpus = 'brown', score_fn = 'dice'):

    print('Conducting Experiment with Context Windows...')
    print('Corpus: {}, Window Size: {}, Score Function: {}'.format(corpus, window_size, score_fn))

    filename = 'experiment_context_window'

    # results_filename = 'results_' + filename + '_' + corpus + '_ws' + str(window_size) + '_' + score_fn + '.csv'

    data = sd.easy_setup_context_window(
        fqt = 10,
        window_size = window_size,
        score_fn = score_fn,
        filename = filename,
        corpus = corpus,
        deserialize = True,
        serialize = True
    )

    words = data['words']
    # Word-Word Co-occurrence Matrix
    WWC = data['WWC']

    word_pairs = util.remove_word_pairs_not_in_corpus(sd.word_pairs, words)

    scores = [
        'nzds',

        'mdcs_cosi',
        'mdcs_seuc',

        'sca_mdcs_cosi',
    ]

    # For each wordpair, calculate ALL the scores!
    word_scores = {}
    for pair in word_pairs:
        for word in pair:
            word = util.normalize([word])[0]
            if not word in word_scores:
                with Timer() as t:
                    word_scores[word] = {}
                    word_scores[word]['nzds'] = sc.nzds(M = WWC, word = word, fns = words)

                    word_scores[word]['mdcs_cosi'] = sc.mdcs(WWC = WWC, word = word, fns = words, metric = 'cosine')
                    word_scores[word]['mdcs_seuc'] = sc.mdcs(WWC = WWC, word = word, fns = words, metric = 'seuclidean')

                    word_scores[word]['sca_mdcs_cosi'] = sc.sca_mdcs(WWC = WWC, word = word, fns = words, metric = 'cosine')
                # print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))
                # print(word_scores[word])

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

    total_hits = [np.sum(results, axis = 0)[i] for i, score in enumerate(scores)]
    percent_hits = [np.sum(results, axis = 0)[i] / len(word_pairs) for i, score in enumerate(scores)]

    results = np.vstack([results, total_hits, percent_hits])

    # for j, score in enumerate(scores):
    #     results[len(word_pairs)] = np.sum(results, axis = 1)
    labels = word_pairs + ['total hits','hit rate']


    # Only give out detailed results in verbose mode.
    if not verbose:
        results = results[-2:,:]
        labels = labels[-2:]

    util.printprettymatrix(M=results, rns = labels, cns = scores)

    return results, labels, scores
    # print(np.sum(results, axis = 0) / results.shape[0], np.sum(results, axis = 0), results.shape[0])