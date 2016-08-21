# -*- coding: utf-8 -*-

import helpers as util
import numpy as np
import pickle

from timer import Timer

from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def retrieve_data_and_tokenize(read_from_file = True):
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

    # Try to read in a previously processed version of the current Corpus.
    # Just to speed things up.
    # If none is found, process it anew!

    return_docs = []
    if read_from_file:
        try:
            file = open("termspec.tmp",'rb')
            return_docs = pickle.load(file)
            file.close()
        except:
            print('File not Found Error while trying to read corpus from drive...')
            print('Continuing with reading it in again')
            read_from_file = False

    if not read_from_file:
        # docs = ['foo bar bar. boom bam bum. foo', 'foo baz bar', 'foo foo baz baz', 'foo', 'bar derp']

        docs = [
            'Optional plotz says to frobnicate the bizbaz first. foo bar bar. foo',
            'Return a foo bang. foo bar. The Ding Dangel Dong Dong foo bang bar.',
            'foo bang baz padauts remoreng.',
            'foo bar bar baz foo'
        ]

        # categories = ['news', 'editorial', 'reviews']
        # categories = ['news']
        # sentences = brown.sents(fileids = brown.fileids(categories=categories))

        # docs = [brown.sents(fileids = fileid) for fileid in brown.fileids(categories=categories)]

        ####################################
        # Compute from brown corpus.

        # count = 0
        # return_docs = []
        # for doc in docs:
        #     returndoc = []
        #     for sent in doc:

        #         sent = util.normalize(sent)
        #         count += len(sent)
        #         returndoc.append(sent)
        #     return_docs.append(returndoc)

        # print(count)

        ###################################
        # Compute from Sample Sentences.

        return_docs = []
        for doc in docs:
            sents = sent_tokenize(doc)
            sents = [word_tokenize(sent) for sent in sents]
            sents = [util.normalize(sent) for sent in sents]
            return_docs.append(sents)


        filehandler = open("termspec.tmp","wb")
        pickle.dump(return_docs,filehandler)
        filehandler.close()


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


    # These are just all the terms for later reference. important!
    # The c indicates that this data structures should be constant: it depends on its ordering, others might use the exact ordering to map term names and values.
    # Python probably has a better data structure for this (tuples)
    c_fns = count_model.get_feature_names()

    # X is now a document-word matrix, but we want a word-word-matrix.
    # The value i,j of the matrix is the count of the word j in document (i.e. sentence) i.
    # Xc ist the word-word-matrix computed from Xc by multiplyig with its transpose.
    Xc = (X.T * X)

    # The Matrix is filled with zero and transformed to the numpy array format.
    M = np.asarray(Xc.todense())

    #TODO: Throw out all-zero vectors!

    # How great is just returning two values? Seriously? python ftw
    return M, c_fns


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


    # These are just all the terms for later reference. important!
    # The c indicates that this data structures should be constant: it depends on its ordering, others might use the exact ordering to map term names and values.
    # Python probably has a better data structure for this (tuples)
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

    # Calculate the Dice Coefficient for each pair of words
    for w_i in range(word_count):
        for w_j in range(word_count):
            dc = 2 * (WWDCM[w_i, w_j]) / (word_counts[w_i] + word_counts[w_j])
            WWDCM[w_i, w_j] = dc

    # util.printprettymatrix(M = WWDCM, cns = fns, rns = fns)

    return WWDCM, fns


def document_word_frequency_matrix(docs):
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


def dfm(M, fns, word):
    """Calculate the df value of a word from a document-word count matrix.

    """
    word_index = fns.index(word)
    # Word count over all documents. It's a Matrix (2d ndarray).
    # May be super duper slow! reevaluate!
    W = M[:, [word_index]]

    # The total number of Documents is just the number of rows of the matrix.
    n_total_documents = W.shape[0]

    # The number of documents where the word appears is the length of the array of nonzero elements in that row
    document_frequency = len(W.nonzero()[0])

    # Scaled document frequency in relation to the total number of documents
    rdfm =  document_frequency / n_total_documents

    return rdfm


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


def tacsm(WWCM, fns, word):
    """Calculates the Total average Cosine similarity Measure for @word.

    TODO: This can be SEVERELY optimized by not calculating the complete cosine matrix first.
    That borders on stupid, really. But no premature optimization until i have a working toy case.

    Arguments:
    WWCM -- Word-Word Cooccurrence Matrix
    fns -- labels for the matrix
    word -- word to calculate the measure for.
    """

    # CSM = cosine_similarity_matrix (WWCM)

    context_vector = WWCM[fns.index(word)]
    nonzero_indices = np.flatnonzero(context_vector)
    # context_terms_names = [fns[i] for i in nonzero_indices]

    # util.printprettymatrix(M = SM, rns = fns, cns = fns)

    # The Subset of WWCM with just the context vector's rows and columns
    # So that the average can be calculated more efficiently.
    # SWWCM = WWCM[:,nonzero_indices][nonzero_indices,:]
    SWWCM = WWCM[nonzero_indices,:]

    # Calculate the cosine distance between each row of SWWCM.
    # Gives a Square nxn Matrix with n = number of rows in SWWCM
    CSM = cosine_similarity_matrix(SWWCM)

    # M[:,[0,2]][[0,2],:]

    # Calculates the Average Cosine distance of all pairs of terms.
    # Does NOT count the main diagonal (distance of each row to itself equals 1).
    # That's what the masking is for.
    mask = np.ones(CSM.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    mean = CSM[mask].mean()

    return mean

def acsm(WWCM, fns, word):
    """Calculates the average Cosine similarity of each context term's cooccurrence vector
    to @word's context vector
    """

    context_vector = WWCM[fns.index(word)]
    nonzero_indices = np.flatnonzero(context_vector)
    SWWCM = WWCM[nonzero_indices,:]
    # print(SWWCM.shape)

    CSM = cdist(SWWCM, np.array([context_vector]), 'cosine', V=None)
    # print(CSM)

    return CSM.mean()




# All the computed values that can be evaluated later on
results = {}

with Timer() as t:
    docs = retrieve_data_and_tokenize(read_from_file = False)
print ('##### Retrieved data and Tokenized in %4.1fs' % t.secs)

sterms = []
sterms.append(util.normalize(['foo'])[0])
sterms.append(util.normalize(['plotz'])[0])

DWFM, c_fns = document_word_frequency_matrix(docs)
WWCM, c_fns = word_word_cooccurrence_matrix(docs)
WWDCM, c_fns = word_word_dice_coeff_matrix(docs)

print('#####################################################################')
print('measure 1: Document Frequency')

print(sterms[0], 1 - dfm(M = DWFM, word = sterms[0], fns = c_fns))
print(sterms[1], 1 - dfm(M = DWFM, word = sterms[1], fns = c_fns))

print('#####################################################################')
print('measure 2: Non Zero Dimensions Measure')

print(sterms[0], 1 - nzdm(M = WWCM, word = sterms[0], fns = c_fns))
print(sterms[1], 1 -nzdm(M = WWCM, word = sterms[1], fns = c_fns))

print('#####################################################################')
print('measure 3: Simple Cooccurrence: Average Cosine Similarity of Context')

print(sterms[0], tacsm(WWCM = WWCM, word=sterms[0], fns=c_fns))
print(sterms[1], tacsm(WWCM = WWCM, word=sterms[1], fns=c_fns))

print('#####################################################################')
print('measure 4: Simple Cooccurrence: Average Cosine Similarity of Context with Term')

print(sterms[0], 1 - acsm(WWCM = WWCM, word=sterms[0], fns=c_fns))
print(sterms[1], 1 - acsm(WWCM = WWCM, word=sterms[1], fns=c_fns))

print('#####################################################################')
print('measure 5: Simple Cooccurrence: Average Cosine Similarity of Context')

print(sterms[0], tacsm(WWCM = WWDCM, word=sterms[0], fns=c_fns))
print(sterms[1], tacsm(WWCM = WWDCM, word=sterms[1], fns=c_fns))

print('#####################################################################')
print('measure 6: Simple Cooccurrence: Average Cosine Similarity of Context with Term')

# Y = cosine_similarity_matrix(X)
print(sterms[0], 1 - acsm(WWCM = WWDCM, word=sterms[0], fns=c_fns))
print(sterms[1], 1 - acsm(WWCM = WWDCM, word=sterms[1], fns=c_fns))