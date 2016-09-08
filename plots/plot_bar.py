# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# data to plot

raw_count_prec = [       0.45     ,  0.21    ]
dice_prec = [    0.74   ,    0.79       ]
labels = [   'mdcs_cosi' , 'mdcs_eucl'  ]
n_groups = len(raw_count_prec)
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, raw_count_prec, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Frequency')
 
rects2 = plt.bar(index + bar_width, dice_prec, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Dice')

plt.plot((0, 3), (0.79,0.79), 'k--', label='document frequency')
plt.plot((0, 3), (0.77,0.77), 'r--', label='nzds')

plt.xlabel('Score')
plt.ylabel('Precision')
plt.title('Sentence Context')
plt.xticks(index + bar_width, labels, )
plt.legend(loc = 4)
 
# plt.tight_layout()
plt.show()