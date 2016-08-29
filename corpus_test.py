# -*- coding: utf-8 -*-


import sys
sys.path.append("termspec/")

import setup_data as sd

import json

print('Starting...')
tokens = sd.retrieve_data_and_tokenize_tokens(corpus = 'tiger')

print(len(tokens), type(tokens))

print(json.dumps(tokens[:1000], indent = 4))




