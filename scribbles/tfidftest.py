# -*- coding: utf-8 -*-

import math
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from nltk.tokenize import sent_tokenize, word_tokenize

# documents = (
# 	"There is no single concept of cool. One ocoolf the essential characteristics of cool is its mutability—what is considered cool changes over time and varies among cultures and generations.",
# 	"One consistent aspect however, is that  is wildly seen as positive and desirable.",
# 	"The sum and substance of cool is a self-conscious aplomb in overall behavior, which entails a set of specific behavioral characteristics that is firmly anchored in symbology, a set of discernible bodily movements, postures, facial expressions and voice modulations that are acquired and take on strategic social value within the peer context.",
# 	"Cool was once an attitude fostered by rebels and underdogs, such as slaves, prisoners, bikers and political dissidents, etc., for whom open rebellion invited punishment, so it hid defiance behind a wall of ironic detachment, distancing itself from the source of authority rather than directly confronting it."

# 	)

documents = ('foo foo bar bar derp derp', 'foo', 'foo')

def tfidf(pdocuments):
	#count term occurrences per document
	documents = []
	all_terms = set()
	for pdocument in pdocuments:
		# tfs are the term frequencies of this document's words.
		document_terms = word_tokenize(pdocument)
		document = { 'tfs': {}, 'tfidfs': {}, 'length':len(document_terms) }
		for term in document_terms:
			all_terms.add(term)
			tfs = document['tfs']
			tfs[term] = tfs[term] + 1 if term in tfs else 1
		documents.append(document)

	# calculate tfidf
	all_terms = sorted(all_terms)
	for term in all_terms:
		N = len(documents)
		df = 0
		for document in documents:
			df += 1 if document['tfs'].get(term) else 0
		for document in documents:
			tf = document['tfs'].get(term) if term in document['tfs'] else 0
			tf = tf/document['length']
			tf_idf = tf * (math.log(N/df) + 1)
			document['tfidfs'][term] = tf_idf
	for document in documents:
		print (sorted(document['tfidfs'].items(), key=lambda x: x[1], reverse=True))

tfidf(documents)

tfidf_vectorizer = TfidfVectorizer(sublinear_tf=False, use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names()
document_tfidf = []

dicto = dict(zip(tfidf_vectorizer.get_feature_names(),tfidf_vectorizer.idf_))
# print(dicto)

for index, feature_name in enumerate(feature_names):
	document_tfidf.append((feature_name, tfidf_matrix[0,index]))

print(sorted(document_tfidf, key=lambda tup: tup[1], reverse=True)[:50])
