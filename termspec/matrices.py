# -*- coding: utf-8 -*-

import helpers as util

from timer import Timer
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from scipy import sparse

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


    # These are just all the terms for later reference. important!
    fns = count_model.get_feature_names()

    # X is now a document-word matrix, but we want a word-word-matrix.
    # The value i,j of the matrix is the count of the word j in document (i.e. sentence) i.
    # Xc ist the word-word-matrix computed from Xc by multiplyig with its transpose.
    Xc = (X.T * X)

    # The Matrix is filled with zero and transformed to the numpy array format.
    M = np.asarray(Xc.todense())

    #TODO: Throw out all-zero vectors!

    # How great is just returning two values? Seriously? python ftw
    return M, fns

def word_word_dice_coeff_matrix_numpy(docs):
    """Calculates DICE coefficient for cooccurrences using numpy. Fast!

    dc = 2*(a & b) / (a * b)
    where a and b are the counts of sentences term_a and term_b appear in, respectively.
    a & b is the count of sentences term_a and term_b both appear in
    """

    # Sentences have to be rejoined. The CountVectorizer only likes strings as document inputs.
    # Then calculates token count per document (in our case each sentence = one document).
    strsents = util.flatten_documents_to_sentence_strings(docs)
    count_model = CountVectorizer(ngram_range=(1,1))
    # Counts each cooccurrence and returns a document-word matrix.
    DWCM_S = count_model.fit_transform(strsents)
    # These are just all the terms for later reference.
    fns = count_model.get_feature_names()

    DWCM_D = np.asarray(DWCM_S.todense())
    # Set to 1 if term occurs in document (ignore multiple occurrences)
    DWCM_D[np.nonzero(DWCM_D)] = 1

    # Converting back and forth between sparse and dense matrizes because:
    # Sparse Matrix Setting is super slow, but dot product fast!
    # Dense Matrix Setting is super fast, but dot product slow!
    # Converting does not seem costly
    DWCM_S = sparse.csr_matrix(DWCM_D)
    WWDC_S = DWCM_S.T * DWCM_S
    WWDC = np.asarray(WWDC_S.todense())

    # with Timer() as t:
    #     # Word-Word Matrix of counts of shared document occurrences (Cooccurences).
    #     # This gives (a & b).
    #     WWDC = np.dot(DWCM.T, DWCM)
    # print('##### ##### Transposed Matrix in %4.1fs' % t.secs)

    # Get the counts of documents each term appears in
    # Count of Sentences per Term
    cospt = DWCM_D.sum(0)

    # Calculate the DICE Coefficient for each word-word pair.
    # WWDC contains the count of documents where both terms cooccure
    # The Array cospt is used normally to get the document count of each i-th term.
    # The Array cospt is transposed to get the document count of each j-th term.
    WWDICE = 2*WWDC / (cospt + cospt[:, np.newaxis])

    return WWDICE, fns

def document_word_frequency_matrix(docs):
    """Computes a document-word occurrence count matrix.

    Occurrence is counted +1 if the term appears in the document.
    """
    strdocs = util.flatten_documents_to_strings(docs)

    count_model = CountVectorizer(ngram_range=(1,1))
    X = count_model.fit_transform(strdocs)
    fns = count_model.get_feature_names()

    # The Matrix is filled with zeros and transformed to the numpy array format.
    X = np.asarray(X.todense())

    return X, fns

def distance_matrix(M, metric = 'cosine'):
    """Computes a SQUARE word-word distance matrix.

    Calculate the distance for each row in the @M with each other.
    I.e. calculate all distances between each vector.

    Arguments:
    M -- Matrix of Context Vectors (Or any others)
    """

    Y = pdist(M, metric = metric)
    # make the cosine distance matrix readable
    Y = squareform(Y)

    return Y

# DEPRECATED
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
    strdocs = util.flatten_documents_to_strings(docs)

    tfidf_model = TfidfVectorizer()

    X = tfidf_model.fit_transform(strdocs)
    fns = tfidf_model.get_feature_names()
    # util.printprettymatrix(X.todense(), cns = fns)

    return X, fn

def word_word_dice_coeff_matrix(docs):
    """Calculates DICE coefficient for cooccurrences.

    dc = 2*(a & b) / (a * b)
    where a and b are the counts of sentences term_a and term_b appear in, respectively.
    a & b is the count of sentences term_a and term_b both appear in
    """

    # Sentences have to be rejoined. The CountVectorizer only likes strings as document inputs.
    # Then calculates token count per document (in our case each sentence = one document).
    strsents = util.flatten_documents_to_sentence_strings(docs)

    count_model = CountVectorizer(ngram_range=(1,1))

    # Counts each cooccurrence and returns a document-word matrix.
    DWCM = count_model.fit_transform(strsents)

    # These are just all the terms for later reference.
    fns = count_model.get_feature_names()

    # In how many sentences does the word appear?
    # Yes, this could be written in a one line comprehension. It would not look good, though.
    word_counts = []
    # util.printprettymatrix(M = DWCM.todense(), cns = fns)
    for word_index, word in enumerate(fns):
        W = DWCM[:, word_index]
        word_counts.append(len(W.nonzero()[0]))
        # print(word, word_counts[word_index])

    word_count = DWCM.shape[1]
    sent_count = DWCM.shape[0]

    with Timer() as t:
        WWDCM = np.zeros(shape=(word_count, word_count))
        # for each word-word-combination check
        for w_i in range(word_count):
            for w_j in range(word_count):
                # for each sentence check
                for s_i in range(sent_count):
                    # Add +1 if both words are in sentence.
                    if not w_i == w_j:
                        if DWCM[s_i, w_i] > 0 and DWCM[s_i, w_j] > 0:
                            WWDCM[w_i, w_j] += 1
                    else:
                        WWDCM[w_i, w_j] = 1
    print('##### ##### Counted Cooccurrences for each Word-Word-Combination in each sentence in %4.1fs' % t.secs)

    # Calculate the Dice Coefficient for each pair of words
    with Timer() as t:
        for w_i in range(word_count):
            for w_j in range(word_count):
                dc = 2 * (WWDCM[w_i, w_j]) / (word_counts[w_i] + word_counts[w_j])
                WWDCM[w_i, w_j] = dc
    print('##### ##### Calculated Dice Coefficient for each Word-Word-Combination in %4.1fs' % t.secs)

    # util.printprettymatrix(M = WWDCM, cns = fns, rns = fns)

    return WWDCM, fns