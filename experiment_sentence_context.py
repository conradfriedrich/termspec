# -*- coding: utf-8 -*-

import sys
sys.path.append("termspec/")

import helpers as util

import scores as sc
import setup_data as sd
import numpy as np
from timer import Timer

#####################################################################
# SETUP

def conduct(verbose = True, corpus = 'brown', score_fn = 'dice'):
    print('Conducting Experiment with Sentence Context...')
    print('Corpus: {}, Score Function: {}'.format(corpus, score_fn))

    filename = 'experiment_sentence_context'
    data = sd.easy_setup_sentence_context(filename = filename, score_fn = score_fn, corpus = 'brown', deserialize = True, serialize = True)

    scores = [
        'occ',
        'dfs',
        'nzds',

        'mdcs_cosi',
        'mdcs_eucl',
        'mdcs_sqeu',
        'mdcs_seuc',

        'sca_mdcs_cosi',
        'sca_mdcs_eucl',
        'sca_mdcs_sqeu',
        'sca_mdcs_seuc',
        ]
    #####################################################################
    # EXPERIMENT


    DWF = data ['DWF']
    WWC = data ['WWC']
    fns = data['fns']

    # reduce word pairs to ones actually appearing in the reduced corpus
    word_pairs = []
    for pair in sd.word_pairs:
        pair_in_words = True
        for word in pair:
            word = util.normalize([word])[0]
            if word not in fns:
                pair_in_words = False
        if pair_in_words:
            word_pairs.append(pair)

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
                    word_scores[word]['dfs'] = sc.dfs(M = DWF, word = word, fns=fns)

                    word_scores[word]['nzds'] = sc.nzds(M = WWC, word = word, fns = fns)

                    word_scores[word]['mdcs_cosi'] = sc.mdcs(WWC = WWC, word = word, fns = fns, metric = 'cosine')
                    word_scores[word]['mdcs_eucl'] = sc.mdcs(WWC = WWC, word = word, fns = fns, metric = 'euclidean')
                    word_scores[word]['mdcs_sqeu'] = sc.mdcs(WWC = WWC, word = word, fns = fns, metric = 'sqeuclidean')
                    word_scores[word]['mdcs_seuc'] = sc.mdcs(WWC = WWC, word = word, fns = fns, metric = 'seuclidean')

                    word_scores[word]['sca_mdcs_cosi'] = sc.sca_mdcs(WWC = WWC, word = word, fns = fns, metric = 'cosine')
                    word_scores[word]['sca_mdcs_eucl'] = sc.sca_mdcs(WWC = WWC, word = word, fns = fns, metric = 'euclidean')
                    word_scores[word]['sca_mdcs_sqeu'] = sc.sca_mdcs(WWC = WWC, word = word, fns = fns, metric = 'sqeuclidean')
                    word_scores[word]['sca_mdcs_seuc'] = sc.sca_mdcs(WWC = WWC, word = word, fns = fns, metric = 'seuclidean')
                # print('##### Calculated scores for %s in  %4.1f' % (word, t.secs))
                # print(word_scores[word])

    #####################################################################
    # RESULTS
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
