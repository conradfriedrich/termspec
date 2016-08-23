# -*- coding: utf-8 -*-

from scipy import sparse
import sys
sys.path.append("termspec/")

import helpers as util

from termspec import core as ts
import numpy as np
from timer import Timer

#####################################################################
# SETUP

filename = 'experiment_c_data.tmp'
data = ts.easy_setup(filename = filename, corpus = 'toy', deserialize = False, serialize = False)


#####################################################################
# EXPERIMENT


DWF = data ['DWF']
WWC = data ['WWC']
WWDICE = data ['WWDICE']
fns = data['fns']

ts.adcs(WWC = WWDICE, fns = fns, word = 'plotz')
