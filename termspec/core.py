# -*- coding: utf-8 -*-

import helpers as util
import numpy as np
import json #just for printing!
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def retrieve_data_and_tokenize():
    """Retrieves the data from the source and makes a neat array of words, sentences and documents out of it.
    [ #docs
        [ #document1
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ],
        [ #document2
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ]
    ]

    """

    # docs = ['foo bar bar. boom bam bum. foo', 'foo baz bar', 'foo foo baz baz', 'foo', 'bar derp']

    docs = [
        'Optional plotz says to frobnicate the bizbaz first. foo bar bar. foo',
        'Return a foo bang. foo bar. The Ding Dangel Dong Dong foo bang bar.',
        'foo bang baz padauts remoreng.',
        'foo bar bar baz foo'
    ]

    # docs = (
    #   "There is no single concept of cool. One of the essential characteristics of cool is its mutability — what is considered cool changes over time and varies among cultures and generations.",
    #   "One consistent aspect however, is that  is wildly seen as positive and desirable.",
    #   "The sum and substance of cool is a self-conscious plomb in overall behavior, which entails a set of specific behavioral characteristics that is firmly anchored in symbology, a set of discernible bodily movements, postures, facial expressions and voice modulations that are acquired and take on strategic social value within the peer context.",
    #   "Cool was once an attitude fostered by rebels and underdogs, such as slaves, prisoners, bikers and political dissidents, etc., for whom open rebellion invited punishment, so it hid defiance behind a wall of ironic detachment, distancing itself from the source of authority rather than directly confronting it."
    #   )


    return_docs = []
    for doc in docs:
        sents = sent_tokenize(doc)
        sents = [word_tokenize(sent) for sent in sents]
        sents = [util.normalize(sent) for sent in sents]
        return_docs.append(sents)

    return return_docs

def word_word_cooccurrence_matrix(docs):
    """Computes a SQUARE cooccurrence matrix of all the terms in the corpus.

    Cooccurence is counted +1 if terms appear in the same sentence.
    """

    # Sentences have to be rejoined. The CountVectorizer only likes strings as document inputs.
    # Then calculates token count per document (in our case each sentence = one document).
    strsents = util.flatten_documents_to_sentence_strings(docs)

    count_model = CountVectorizer(ngram_range=(1,1))

    # Counts each cooccurrence and returns a document-word matrix.
    X = count_model.fit_transform(strsents)

    # X is now a document-word matrix, but we want a word-word-matrix.
    # The value i,j of the matrix is the count of the word j in document (i.e. sentence) i.
    # Xc ist the word-word-matrix computed from Xc by multiplying with its transpose
    Xc = (X.T * X)

    # Main diagonal to 0, useless anyway.
    # Xc.setdiag(0)

    # The Matrix is filled with zero and transformed to the numpy array format.
    M = np.asarray(Xc.todense())

    # These are just all the terms for later reference. important!
    # The c indicates that this data structures should be constant: it depends on its ordering, others might use the exact ordering to map term names and values.
    # Python probably has a better data structure for this (tuples)
    c_fns = count_model.get_feature_names()

    #TODO: Throw out all-zero vectors!

    # How great is just returning two values? Seriously? python ftw
    return M, c_fns

def document_word_count_matrix(docs):
    """Computes a document-word occurrence count matrix.

    Occurrence is counted +1 if the term appears in the document.
    """
    strdocs = util.flatten_documents_to_strings(docs)

    count_model = CountVectorizer(ngram_range=(1,1))
    X = count_model.fit_transform(strdocs)
    c_fns = count_model.get_feature_names()

    # The Matrix is filled with zeros and transformed to the numpy array format.
    X = np.asarray(X.todense())

    return X, c_fns

def cosine_similarity_matrix(M):
    """Computes a SQUARE word-word cosine distance matrix.

    Calculate the cosine distance for each row in the @M with each other.
    I.e. calculate all cosine distances between each vector.

    Arguments:
    M -- Matrix of Context Vectors (Or any others)
    """

    Y = pdist(M, metric='cosine')
    # make the cosine distance matrix readable
    Y = squareform(Y)

    # #take not the cosine distance, but similarity (applied per cell)
    Y = 1 - Y

    return Y

def tfidf_matrix(docs):
    """Computes a tfidf matrix for the given corpus.

    Arguments:
    docs -- expects docs in the format
    [ #docs
        [ #document1
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ],
        [ #document2
            ['word1','word2','word3'], #sentence1
            ['word1','word2','word3'], #sentence2
        ]
    ]

    Returns:
    Matrix of tfidf values,
    List of feature names
    """
    # Docs have to be rejoined. The TfidfVectorizer only likes strings as document inputs.
    # Then calculates tfidf per term in document
    strdocs = util.flatten_documents_to_strings

    tfidf_model = TfidfVectorizer()

    X = tfidf_model.fit_transform(strdocs)
    c_fns = tfidf_model.get_feature_names()
    # util.printprettymatrix(X.todense(), cns = c_fns)

    return X, c_fns



def df(M, fns, word):
    """Calculate the df value of a word from a document-word count matrix.

    """
    word_index = fns.index(word)
    # Word count over all documents. It's a Matrix (2d ndarray).
    W = M[:, [word_index]]

    # The total number of Documents is just the number of rows of the matrix.
    n_total_documents = W.shape[0]

    # The number of documents where the word appears is the length of the array of nonzero elements in that row
    document_frequency = len(W.nonzero()[0])

    # Scaled document frequency in relation to the total number of documents
    sdf =  document_frequency / n_total_documents

    return sdf

def nzdm(M, fns, word):
    """Calculates the non zero dimensional measure for @word

    Calculates the count of total unique cooccurences for the given word divided by the total of words.
    The result ist the percentage of the words that @word stands in cooccurence with.
    """

    context_vector = M[c_fns.index(word)]
    n_total_dimensions = len(c_fns)
    n_non_zero_dimensions = len(context_vector.nonzero()[0])

    non_zero_dimensions_measure = n_non_zero_dimensions / n_total_dimensions
    return non_zero_dimensions_measure

def csm(CM, SM, fns, word):
    """Calculates the Cosine similarity Measure for @word.

    TODO: This can be SEVERELY optimized by not calculating the complete cosine matrix first.
    That borders on stupid, really. But no premature optimization until i have a working toy case.

    Arguments:
    CM -- Cooccurrence Matrix
    SM -- Similarity Matrix
    fns -- labels for the matrix
    word -- word to calculate the measure for.
    """


    context_vector = CM[fns.index(word)]
    nonzero_indices = np.flatnonzero(context_vector)
    context_terms_names = [fns[i] for i in nonzero_indices]

    # util.printprettymatrix(M = SM, rns = fns, cns = fns)

    # The Subset of SM with just the context vector's rows and columns 
    # So that the average can be calculated

    SSM = SM[:,nonzero_indices][nonzero_indices,:]
    # M[:,[0,2]][[0,2],:]
    # print(SM)

    # Calculates the Average Cosine distance of all pairs of terms
    mask = np.ones(SSM.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    mean = SSM[mask].mean()

    return mean

# All the computed values that can be evaluated later on
results = {}

docs = retrieve_data_and_tokenize()

#####################################################################

print('#####################################################################')
print('measure 1: Document Frequency', 'Higher means less specific')
X, c_fns = document_word_count_matrix(docs)
# util.printprettymatrix(M = X, cns = c_fns)



sterm = 'plotz'
print(sterm, 1 - df(M = X, word = sterm, fns = c_fns))
sterm = 'foo'
print(sterm, 1 - df(M = X, word = sterm, fns = c_fns))

#####################################################################

print('#####################################################################')
print('measure 2: Non Zero Dimensions Measure', 'Higher means less specific')
X, c_fns = word_word_cooccurrence_matrix(docs)

sterm = 'plotz'
print(sterm, 1 - nzdm(M = X, word = sterm, fns = c_fns))
sterm = 'foo'
print(sterm, 1 - nzdm(M = X, word = sterm, fns = c_fns))


#####################################################################

print('#####################################################################')
print('measure 3: Average Cosine Similarity', 'Higher means MORE specific')

Y = cosine_similarity_matrix(X)

sterm = 'plotz'
print(sterm, csm(CM = X, SM = Y, word=sterm, fns=c_fns))
sterm = 'foo'
print(sterm, csm(CM = X, SM = Y, word=sterm, fns=c_fns))




