# -*- coding: utf-8 -*-


from nltk.corpus import brown
import spinfoutils as utils

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

categories = ['news', 'editorial', 'reviews']
newsfileids = brown.fileids(categories=categories)
sentences = brown.sents(fileids = brown.fileids(categories=categories))

documents = [brown.sents(fileids = fileid) for fileid in brown.fileids(categories=categories)]

tfidf_vectorizer = TfidfVectorizer(
	sublinear_tf = True,
	use_idf = True,
	smooth_idf=True
	)

# strdocuments = []
# for document in documents:
# 	strsentences = []
# 	for sentence in document:
# 		strsentences.append(' '.join(sentence))
# 	strdocuments.append(' '.join(strsentences))

#convert the documentslist of listf of sentences (which are list of words) into a list of full-document-strings
strdocumentscomp = [' '.join([' '.join(i) for i in document]) for document in documents]

tfidf_matrix = tfidf_vectorizer.fit_transform(strdocumentscomp)
print(tfidf_matrix.shape)
feature_names = tfidf_vectorizer.get_feature_names()
document_tfidf = []
for index, feature_name in enumerate(feature_names):
	document_tfidf.append((feature_name, tfidf_matrix[1,index]))

print(sorted(document_tfidf, key=lambda tup: tup[1], reverse=True)[:50])


