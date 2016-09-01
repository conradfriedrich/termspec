# -*- coding: utf-8 -*-

import experiment_context_window as ecw
import experiment_sentence_context as esc
import json

corpus = 'brown'
# corpus = 'reuters'

# score_fns = ['raw_count', 'dice']

# for score_fn in score_fns:
#     esc.conduct(verbose = False, corpus = corpus, score_fn = score_fn)


score_fns = ['binary', 'raw_count', 'chi_sq', 'dice']
# score_fns = ['raw_count']

score_lists = {}

for window_size in [2,3,4,5,8,10,15,20,25,30,35,40,46,52,60,80,100]:

# for window_size in [90,100,110]:
    for score_fn in score_fns:

        if not score_fn in score_lists:
            score_lists[score_fn] = {}

        results, labels, scores = ecw.conduct(verbose = False, corpus = corpus, window_size = window_size, score_fn = score_fn)
        for i, score in enumerate(scores):
            if not score in score_lists[score_fn]:
                score_lists[score_fn][score] = []
            score_lists[score_fn][score].append(results[1,i])

print(json.dumps(score_lists, sort_keys=True, indent = 4))