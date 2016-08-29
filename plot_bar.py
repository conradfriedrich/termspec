# -*- coding: utf-8 -*-






# fig = plt.gcf()
# plot_url = py.plot_mpl(fig, filename='mpl-basic-bar')

import numpy as np
import matplotlib.pyplot as plt

# data to plot

raw_count_prec = [0.77,0.79 ,0.77,0.45,0.21,0.2,0.73,0.43,0.73,0.71,0.76]
dice_prec = [0.77,0.79,0.77,0.74,0.79,0.77,0.77,0.4,0.25,0.24,0.46]
labels = ['occ','dfs','nzds','mdcs_cosi','mdcs_eucl','mdcs_sqeu','mdcs_seuc','sca_mdcs_cosi','sca_mdcs_eucl','sca_mdcs_sqeu','sca_mdcs_seuc']

n_groups = len(raw_count_prec)
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, raw_count_prec, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Raw Count')
 
rects2 = plt.bar(index + bar_width, dice_prec, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Dice')
 
plt.xlabel('Score')
plt.ylabel('Precision Percentage')
plt.title('Brown Corpus: Precision of different Scores in Sentence Contexts')
plt.xticks(index + bar_width, labels)
plt.legend()
 
plt.tight_layout()
plt.show()