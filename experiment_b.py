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

def conduct(verbose = True, window_size = 4, corpus = 'brown', score_fn = 'dice'):
    print('Conducting Experiment with Sentence Context...')
    print('Corpus: {}, Window Size: {}, Score Function: {}'.format(corpus, window_size, score_fn))

    filename = 'experiment_b_data'
    data = sd.easy_setup_sentence_context(filename = filename, score_fn = score_fn, corpus = 'brown', deserialize = True, serialize = True)


    scores = [
        'occ',
        'dfs',
        'nzds',

        # 'sc_c_mdcs',
        # 'sc_e_mdcs',
        # 'sc_se_mdcs',

        # 'sc_c_mdcs_mc',
        # 'sc_e_mdcs_mc',
        # 'sc_se_mdcs_mc',

        # 'sc_c_mdcs_sca',
        # 'sc_e_mdcs_sca',
        # 'sc_se_mdcs_sca',

        # 'dc_c_mdcs',
        # 'dc_e_mdcs',
        # 'dc_se_mdcs',

        # 'dc_c_mdcs_mc',
        # 'dc_e_mdcs_mc',
        # 'dc_se_mdcs_mc',

        # 'dc_c_mdcs_sca',
        # 'dc_e_mdcs_sca',
        # 'dc_se_mdcs_sca',
        ]
    #####################################################################
    # EXPERIMENT


    DWF = data ['DWF']
    WWC = data ['WWC']
    WWDICE = data ['WWDICE']
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

                    # word_scores[word]['sc_c_mdcs'] = sc.mdcs(WWC = WWC, word = word, fns = fns, metric = 'cosine')
                    # word_scores[word]['sc_e_mdcs'] = sc.mdcs(WWC = WWC, word = word, fns = fns, metric = 'euclidean')
                    # word_scores[word]['sc_se_mdcs'] = sc.se_mdcs(WWC = WWC, word = word, fns = fns)

                    # word_scores[word]['sc_c_mdcs_mc'] = sc.mdcs_mc(WWC = WWC, mc = 200, word = word, fns = fns, metric = 'cosine')
                    # word_scores[word]['sc_e_mdcs_mc'] = sc.mdcs_mc(WWC = WWC, mc = 200, word = word, fns = fns, metric = 'euclidean')
                    # word_scores[word]['sc_se_mdcs_mc'] = sc.se_mdcs_mc(WWC = WWC, mc = 200, word = word, fns = fns)

                    # word_scores[word]['sc_c_mdcs_sca'] = sc.mdcs_sca(WWC = WWC, word = word, fns = fns, metric = 'cosine')
                    # word_scores[word]['sc_e_mdcs_sca'] = sc.mdcs_sca(WWC = WWC, word = word, fns = fns, metric = 'euclidean')
                    # word_scores[word]['sc_se_mdcs_sca'] = sc.se_mdcs_sca(WWC = WWC, word = word, fns = fns)

                    # word_scores[word]['dc_c_mdcs'] = sc.mdcs(WWC = WWDICE, word = word, fns = fns, metric = 'cosine')
                    # word_scores[word]['dc_e_mdcs'] = sc.mdcs(WWC = WWDICE, word = word, fns = fns, metric = 'euclidean')
                    # word_scores[word]['dc_se_mdcs'] = sc.se_mdcs(WWC = WWDICE, word = word, fns = fns)

                    # word_scores[word]['dc_c_mdcs_mc'] = sc.mdcs_mc(WWC = WWDICE, mc = 50, word = word, fns = fns, metric = 'cosine')
                    # word_scores[word]['dc_e_mdcs_mc'] = sc.mdcs_mc(WWC = WWDICE, mc = 200, word = word, fns = fns, metric = 'euclidean')
                    # word_scores[word]['dc_se_mdcs_mc'] = sc.se_mdcs_mc(WWC = WWDICE, mc = 200, word = word, fns = fns)

                    # word_scores[word]['dc_c_mdcs_sca'] = sc.mdcs_sca(WWC = WWDICE, word = word, fns = fns, metric = 'cosine')
                    # word_scores[word]['dc_e_mdcs_sca'] = sc.mdcs_sca(WWC = WWDICE, word = word, fns = fns, metric = 'euclidean')
                    # word_scores[word]['dc_se_mdcs_sca'] = sc.se_mdcs_sca(WWC = WWDICE, word = word, fns = fns)

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


    # for j, score in enumerate(scores):
    #     results[len(word_pairs)] = np.sum(results, axis = 1)

    util.printprettymatrix(M=results, rns = word_pairs, cns = scores)

    print(np.sum(results, axis = 0) / results.shape[0])

