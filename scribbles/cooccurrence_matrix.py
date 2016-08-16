# -*- coding: utf-8 -*-

import spinfoutils as util
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer

def retrieve_data_and_tokenize():

	docs = ['foo bar bar', 'foo baz bar', 'foo foo baz baz', 'foo', 'bar derp']

	# docs = (
	# 	"There is no single concept of cool. One of the essential characteristics of cool is its mutability — what is considered cool changes over time and varies among cultures and generations.",
	# 	"One consistent aspect however, is that  is wildly seen as positive and desirable.",
	# 	"The sum and substance of cool is a self-conscious aplomb in overall behavior, which entails a set of specific behavioral characteristics that is firmly anchored in symbology, a set of discernible bodily movements, postures, facial expressions and voice modulations that are acquired and take on strategic social value within the peer context.",
	# 	"Cool was once an attitude fostered by rebels and underdogs, such as slaves, prisoners, bikers and political dissidents, etc., for whom open rebellion invited punishment, so it hid defiance behind a wall of ironic detachment, distancing itself from the source of authority rather than directly confronting it."
	# 	)
	#sentential tokenization. the cooccurence counting at the moment is just sentence based.
	sents = []
	for doc in docs:
		sents.extend(sent_tokenize(doc))

	#tokenize and normalize the sentences
	sents = [word_tokenize(sent) for sent in sents]
	sents = [util.normalize(sent) for sent in sents]

	return sents

def compute_cooccurrence_matrix(sents):

	#sentences have to be rejoined. The CountVectorizer only likes strings as document inputs. Then calculates token count per document (in our case each sentence = one document)
	docs = [' '.join(sent) for sent in sents]

	count_model = CountVectorizer(ngram_range=(1,1)) # default unigrams
	X = count_model.fit_transform(docs)
	Xc = (X.T * X) # woot matrix magic, how does this work?
	Xc.setdiag(0) # maindiagonal to 0, useless anyway!
	M = np.asarray(Xc.todense())
	c_fns = count_model.get_feature_names()

	return M, c_fns

# the c indicates that this data structures should be constant: it depends on its ordering, others might use the exact ordering to map term names and values
sents = retrieve_data_and_tokenize()
M, c_fns = compute_cooccurrence_matrix(sents)

print(c_fns)
util.printprettymatrix(M, c_fns)

# find the context for sterm with cooccurence != 0
sterm = 'foo'
# this is sterm's row of the cooccurence matrix
c_context_vector = M[c_fns.index(sterm)]

#remove all zero elements
nonzero_indices = np.flatnonzero(c_context_vector)

#retrieve all context terms that stand in actual cooccurrence with sterm
context_terms = [c_fns[i] for i in nonzero_indices]
print(context_terms)

# calculate the cosine distance for each row in the context matrix with each other
# i.e. calculate all cosine distances between each cooccurrence vector.
Y = pdist(M,metric='cosine')
# make the cosine distance matrix readable
Y = squareform(Y)
# #take not the cosine distance, but similarity (applied per cell)
Y = 1 - Y
util.printprettymatrix(Y, c_fns)

