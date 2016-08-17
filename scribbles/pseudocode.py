

corpus = get_englisch_corpus
corpus = remove_stopwords_and_punctuations(corpus)
corpus = normalize(corpus)
#corpus should be an array of documents.
#each document should be an array of words (sentences).

corpus.get_first_document()
>>> ['attitude','guy','lie',...]

cooccurrences = analyse_cooccurrence('cool',corpus.get_first_document())
>>> [('attitude',3),('guy',6),('lie',0),...]

print cooccurrences['attitude']
>>> 3

# reduce dimensionality thorugh tfidf analysis

#build a cooccurence matrix. ist that necessary every time? i dont think so, but lets see.

# first score: measure the document frequency of the pair of terms to determine which one is more specific. the higher the document frequency, the more general and therefore less specific the term.

#second score: determine the terms' cooccurrences. keep in mind the scoring function for the cooccurrence as changing it might yield different results. then define and calculate a score raising with the amount of different cooccurences and lowering with the total amount of terms in the corpus. with what percentage of terms doesgiven term cooccure?

# third score: the average cosine similarity of the cooccurrence's contect vectors (acs-index). For each term in the analysans' cooccurences, coined context, determine it's cooccurrences. then, take the cosine similarity between each of thecontext's term's context vector. the average of those similarity should yield somemeasure for the spread of context. See Dunn Index, Rand Index, Huberts gamma coefficient. Dunn Index might be worth to look into for these purposes.

#proposed advantage of acs: the df model is highly dependent on the type on corpus that is used. if a word appears in a corpus some given times and you then add a lot of other documents without the word to it, the document frequence drastically declines. not so with the cosine measure: it is independet of added null-dimensions. problem with the context, as it gets larger and the added documents, although not containing the original word, may change the acs index.

# possibility to test: use HYPERONYMS from wordnet. Hyperonyms are hierarchies of terms, with the more general terms higher up in the hierarchy. Can the score correctly evaluate the lower/higher specificity? Compare predefined examples, calculate average Success rate. Is the third score any more effective than the first?

# Finally: Choose the best method / score to evaluate anglicisms in german / english. Are they comparably specific or do they significantly differ in specificity?

# Could be interesting to do variations on the cooccurrence to see wether something changes

#Is the cosinus similarity constant for given vectors if you add nulldimensions?
#!important for comparing different vector spaces
# -> ja (siehe cosinetest.py)