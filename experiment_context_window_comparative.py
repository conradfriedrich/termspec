# -*- coding: utf-8 -*-




import sys
sys.path.append("termspec/")

import helpers as util
import setup_data as sd
import scores as sc

import numpy as np

def conduct(verbose = True, window_size = 4, score_fn = 'dice'):
    """Conducts an Experiment that compares english and german anglicisms for the given @window_size.

    Prints out the Result to std.out
    """
    print('Conducting Experiment with Anglicisms in Context Window...')
    print('Window Size: {}, Score Function: {}'.format(window_size, score_fn))

    anglicisms = sd.anglicisms

    data_eng = sd.easy_setup_context_window(
            fqt = 10,
            window_size = window_size,
            score_fn = score_fn,
            corpus = 'reuters',
            deserialize = True,
            serialize = True
        )
    words_eng = data_eng['words']
    # Word-Word Co-occurrence Matrix
    WWC_eng = data_eng['WWC']
    print('eng words', len(words_eng))
    anglicisms = util.remove_words_not_in_list(anglicisms, words_eng, language = 'english')

    data_ger = sd.easy_setup_context_window(
            fqt = 10,
            window_size = window_size,
            score_fn = score_fn,
            corpus = 'tiger',
            deserialize = True,
            serialize = True
        )
    words_ger = data_ger['words']
    print('ger words', len(words_ger))

    # print(json.dumps(words_ger, indent = 4))
    # Word-Word Co-occurrence Matrix
    WWC_ger = data_ger['WWC']

    anglicisms = util.remove_words_not_in_list(anglicisms, words_ger, language = 'german')

    scores = ['nzds','mdcs_cosi', 'mdcs_seuc', 'sca_mdcs_cosi']
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
        # print(word, word_scores[word])
    results = np.zeros( (len(anglicisms),len(scores)), dtype=bool)

    for i, word in enumerate(anglicisms):

        for j, score in enumerate(scores):
            # Now this says TRUE or 1.0 if the english term is more general than the german
            results[i,j] = word_scores[word][score][0] > word_scores[word][score][1]


    util.printprettymatrix(M=results, rns = anglicisms, cns = scores)
