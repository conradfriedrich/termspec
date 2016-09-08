# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
from random import randint
import matplotlib.patches as mpatches



compact_x = [randint(5,40) for _ in range(60)]
compact_y = [randint(5,40) for _ in range(60)]
spread_x = [randint(13,100) for _ in range(40)]
spread_y = [randint(22,88) for _ in range(40)]

centroid_compact_x = np.mean(compact_x)
centroid_compact_y = np.mean(compact_y)

centroid_spread_x = np.mean(spread_x)
centroid_spread_y = np.mean(spread_y)


plt.plot(compact_x, compact_y, 'r^', label='Kompakter Kontext' )
plt.plot(centroid_compact_x, centroid_compact_y, 'ro', label='Centroid Kompakt', markersize='15' )
plt.plot(spread_x, spread_y, 'gs', label='Stark gestreuter Kontext' )
plt.plot(centroid_spread_x, centroid_spread_y, 'go', label='Centroid Gestreut', markersize='15' )


plt.axis([0,100,0,100])
plt.legend(numpoints = 1, loc = 0)

plt.show()
# x = [2,3,4,5,8,10,15,20,22,24,25,26,28,30,34,36,40,46,52,60]


# plt.plot(x, data['raw_count']['nzds'], 'r^-', label='nzds' )
# plt.plot(x, data['raw_count']['mdcs_cosi'], 'bs-', label='mdcs_cosi')
# plt.plot(x, data['raw_count']['mdcs_seuc'], 'yo-', label='mdcs_seuc')
# plt.plot(x, data['raw_count']['sca_mdcs_cosi'], 'cv-', label='sca_mdcs_cosi' )
# plt.plot((0, 62), (0.79,0.79), 'k--', label='document frequency')
# plt.axis([0,62,0.5,1.0])
# plt.xlabel('Size of Context Window')
# plt.ylabel('Precision Percentage')
# plt.title('{} Corpus: Matrix computed with Raw Count'.format(corpus))
# plt.legend()
# plt.show()

# plt.plot(x, data['chi_sq']['nzds'], 'r^-', label='nzds' )
# plt.plot(x, data['chi_sq']['mdcs_cosi'], 'bs-', label='mdcs_cosi')
# plt.plot(x, data['chi_sq']['mdcs_seuc'], 'yo-', label='mdcs_seuc')
# plt.plot(x, data['chi_sq']['sca_mdcs_cosi'], 'cv-', label='sca_mdcs_cosi' )
# plt.plot((0, 62), (0.79,0.79), 'k--', label='document frequency')
# plt.axis([0,62,0.5,1.0])
# plt.xlabel('Size of Context Window')
# plt.ylabel('Precision Percentage')
# plt.title('{} Corpus: Matrix computed with Chi Squared'.format(corpus))
# plt.legend()
# plt.show()

# plt.plot(x, data['dice']['nzds'], 'r^-', label='nzds')
# plt.plot(x, data['dice']['mdcs_cosi'], 'bs-', label='mdcs_cosi')
# plt.plot(x, data['dice']['mdcs_seuc'], 'yo-', label='mdcs_seuc')
# plt.plot(x, data['dice']['sca_mdcs_cosi'], 'cv-', label='sca_mdcs_cosi')
# plt.plot((0, 62), (0.79,0.79), 'k--', label='document frequency')
# plt.axis([0,62,0.5,1.0])
# plt.xlabel('Size of Context Window')
# plt.ylabel('Precision Percentage')
# plt.title('{} Corpus: Matrix computed with Dice Coefficient'.format(corpus))
# plt.legend()
# plt.show()








