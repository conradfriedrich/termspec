# -*- coding: utf-8 -*-

import experiment_context_window as ecw
import experiment_sentence_context as esc

corpus = 'brown'

score_fns = ['raw_count', 'raw_freq', 'phi_sq', 'chi_sq', 'dice']

for window_size in [3,4,5,6,10,20]:
    for score_fn in score_fns:
        ecw.conduct(verbose = False, corpus = corpus, window_size = window_size, score_fn = score_fn)

score_fns = ['raw_count', 'dice']

for score_fn in score_fns:
    esc.conduct(verbose = False, corpus = corpus, score_fn = score_fn)