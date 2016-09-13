# -*- coding: utf-8 -*-

"""Script to run the experiment for anglicisms with different parameters"""


import experiment_context_window_comparative as ecwc

score_fns = ['binary', 'raw_count', 'chi_sq', 'dice']

score_lists = {}

for window_size in [4,25,60,100]:

# for window_size in [90,100,110]:
    for score_fn in score_fns:

        if not score_fn in score_lists:
            score_lists[score_fn] = {}

        ecwc.conduct(verbose = False, window_size = window_size, score_fn = score_fn)
        # for i, score in enumerate(scores):
        #     if not score in score_lists[score_fn]:
        #         score_lists[score_fn][score] = []
        #     score_lists[score_fn][score].append(results[1,i])

# print(json.dumps(score_lists, sort_keys=True, indent = 4))