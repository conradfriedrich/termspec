# -*- coding: utf-8 -*-

import experiment_context_window as ecw
import experiment_sentence_context as esc
import json

corpus = 'tiger'

score_fns = ['raw_count', 'chi_sq', 'dice']

score_lists = {}

for window_size in [4,10]:
    for score_fn in score_fns:

        if not score_fn in score_lists:
            score_lists[score_fn] = {}

        results, labels, scores = ecw.conduct(language = 'german', verbose = True, corpus = corpus, window_size = window_size, score_fn = score_fn)
        for i, score in enumerate(scores):
            if not score in score_lists[score_fn]:
                score_lists[score_fn][score] = []
            score_lists[score_fn][score].append(results[1,i])

print(json.dumps(score_lists, sort_keys=True, indent = 4))