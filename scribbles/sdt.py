# -*- coding: utf-8 -*-

import numpy as np

np.set_printoptions(suppress=True)
# Sehr erhellend, was hier üpassiert!
M = np.ones((3,3))
MEAN = [5,100,1]
SD = M - np.array(MEAN)[:, np.newaxis]


M = np.array([   [273, 1601, 3],
                    [123, 1313, 6],
                    [992, 1235, 8]    ])

print(M)



print(VARIANCE)