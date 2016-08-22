# -*- coding: utf-8 -*-

import helpers as util
import numpy as np
import pickle

from timer import Timer

from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def easy_setup(filename = None, corpus = 'toy', deserialize = True, serialize = True):
    """Sets up data data object for experiments.

    If a filename is given, tries to deserialize from that file.
    Creates documents from the given corpus,
    a Document Word Frequency Matrix,
    a Word Word Cooccurrence Matrix,
    and a Word Word Dice Coefficient Matrix.

    Also writes the result to file to speed up the next run.
    """

    data = None

    if filename and deserialize:
        with Timer() as t:
            data = {}
            data['docs'] = util.read_from_file(filename + '_docs')
            data['DWF'] = util.read_from_file(filename + '_DWF')

            # The large matrizes have been serialized in sparse format.
            WWC_S = util.read_from_file(filename + '_WWC')
            WWDICE_S = util.read_from_file(filename + '_WWDICE')
            data['WWC'] = np.asarray(WWC_S.todense())
            data['WWDICE'] = np.asarray(WWDICE_S.todense())
            data['fns'] = util.read_from_file(filename + '_fns')

        print ('##### Deserialized Data in %4.1fs' % t.secs)

    if not data:
        data = {}
        with Timer() as t:
            data['docs'] = retrieve_data_and_tokenize(corpus = corpus)
        print ('##### Retrieved data and Tokenized in %4.1fs' % t.secs)

        with Timer() as t:
            data['DWF'], fns = document_word_frequency_matrix(data['docs'])
        print('##### Created Document Word Frequency Matrix in %4.1fs' % t.secs)

        with Timer() as t:
            data['WWC'], fns = word_word_cooccurrence_matrix(data['docs'])
        print('##### Created Word Word Cooccurrence Matrix in %4.1fs' % t.secs)

        with Timer() as t:
            data['WWDICE'], fns = word_word_dice_coeff_matrix_numpy(data['docs'])
        print('##### Created Word Word Dice Coefficient Matrix in %4.1fs' % t.secs)

        data ['fns'] = fns

        with Timer() as t:
            if filename and serialize:
                util.write_to_file(filename + '_docs', data['docs'])
                util.write_to_file(filename + '_DWF', data['DWF'])
                util.write_to_file(filename + '_fns', data['fns'])

                # Let's save about 99% of disk space.
                WWC_S = sparse.csr_matrix(data['WWC'])
                WWDICE_S = sparse.csr_matrix(data['WWDICE'])

                util.write_to_file(filename + '_WWC', WWC_S)
                util.write_to_file(filename + '_WWDICE', WWDICE_S)
        print ('##### Serialized Data in %4.1fs' % t.secs)

    return data

def retrieve_data_and_tokenize(corpus = 'toy'):
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

    if corpus == 'toy':
    # docs = ['foo bar bar. boom bam bum. foo', 'foo baz bar', 'foo foo baz baz', 'foo', 'bar derp']

        docs = [
            'Optional plotz says to frobnicate the bizbaz first. foo bar bar. foo',
            'Return a foo bang. foo bar. The Ding Dangel Dong Dong foo bang bar.',
            'foo bang baz padauts remoreng.',
            'foo bar bar baz foo'
        ]

        ###################################
        # Compute from Sample Sentences.

        return_docs = []
        for doc in docs:
            sents = sent_tokenize(doc)
            sents = [word_tokenize(sent) for sent in sents]
            sents = [util.normalize(sent) for sent in sents]
            return_docs.append(sents)

    elif corpus == 'brown':

        categories = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
        # categories = ['news', 'editorial', 'reviews']
        # categories = ['news', 'editorial']
        # categories = ['news']
        # sentences = brown.sents(fileids = brown.fileids(categories=categories))

        docs = [brown.sents(fileids = fileid) for fileid in brown.fileids(categories=categories)]

        ##################################
        # Compute from brown corpus.

        count = 0
        return_docs = []
        for doc in docs:
            returndoc = []
            for sent in doc:

                sent = util.normalize(sent)
                count += len(sent)
                returndoc.append(sent)
            return_docs.append(returndoc)

        print(count)

    else:
        raise ValueError('Corpus passed is not known.', corpus)

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

def word_word_dice_coeff_matrix_numpy(docs):
    """Calculates DICE coefficient for cooccurrences using numpy. Fast!

    dc = 2*(a & b) / (a * b)
    where a and b are the counts of sentences term_a and term_b appear in, respectively.
    a & b is the count of sentences term_a and term_b both appear in
    """
    print('##### ##### Started to calculate NUMPY Dice Coefficient')

    with Timer() as t:
        # Sentences have to be rejoined. The CountVectorizer only likes strings as document inputs.
        # Then calculates token count per document (in our case each sentence = one document).
        strsents = util.flatten_documents_to_sentence_strings(docs)
        count_model = CountVectorizer(ngram_range=(1,1))
        # Counts each cooccurrence and returns a document-word matrix.
        DWCM_S = count_model.fit_transform(strsents)
        # These are just all the terms for later reference.
        fns = count_model.get_feature_names()
    print('##### ##### Got Document Word Cooccurrence Matrix in %4.1fs' % t.secs)

    with Timer() as t:
        DWCM_D = np.asarray(DWCM_S.todense())
        # Set to 1 if term occurs in document (ignore multiple occurrences)
        DWCM_D[np.nonzero(DWCM_D)] = 1
    print('##### ##### Set Nonzero values to 1 in %4.1fs' % t.secs)

    # Converting back and forth between sparse and dense matrizes because:
    # Sparse Matrix Setting is super slow, but dot product fast!
    # Dense Matrix Setting is super fast, but dot product slow!
    # Converting does not seem costly
    with Timer() as t:
        DWCM_S = sparse.csr_matrix(DWCM_D)
        WWDC_S = DWCM_S.T * DWCM_S
        WWDC = np.asarray(WWDC_S.todense())
    print('##### ##### Fun with sparse matrizes in %4.1fs' % t.secs)


    # with Timer() as t:
    #     # Word-Word Matrix of counts of shared document occurrences (Cooccurences).
    #     # This gives (a & b).
    #     WWDC = np.dot(DWCM.T, DWCM)
    # print('##### ##### Transposed Matrix in %4.1fs' % t.secs)

    with Timer() as t:
        # Get the counts of documents each term appears in
        # Count of Sentences per Term
        cospt = DWCM_D.sum(0)
    print('##### ##### Prepared to calculate NUMPY Dice Coefficient in %4.1fs' % t.secs)

    # Calculate the DICE Coefficient for each word-word pair.
    # WWDC contains the count of documents where both terms cooccure
    # The Array cospt is used normally to get the document count of each i-th term.
    # The Array cospt is transposed to get the document count of each j-th term.
    with Timer() as t:
        WWDICE = 2*WWDC / (cospt + cospt[:, np.newaxis])
    print('##### ##### Calculated NUMPY Dice Coefficient for each Word-Word-Combination in %4.1fs' % t.secs)

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

    # # #take not the distance, but similarity (applied per cell)
    # Y = 1 - Y

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
    strdocs = util.flatten_documents_to_strings(docs)

    tfidf_model = TfidfVectorizer()

    X = tfidf_model.fit_transform(strdocs)
    fns = tfidf_model.get_feature_names()
    # util.printprettymatrix(X.todense(), cns = fns)

    return X, fns


def dfs(M, fns, word):
    """Calculate the Document Frequency Score of a word from a document-word count matrix.

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


def nzds(M, fns, word):
    """Calculates the Non Zero Dimensions Score for @word

    Calculates the count of total unique cooccurences for the given word divided by the total of words.
    The result ist the percentage of the words that @word stands in cooccurence with.
    """

    context_vector = M[fns.index(word)]
    n_total_dimensions = len(fns)
    n_non_zero_dimensions = len(context_vector.nonzero()[0])

    non_zero_dimensions_measure = n_non_zero_dimensions / n_total_dimensions
    return non_zero_dimensions_measure


def tacds(WWC, fns, word, metric = 'cosine'):
    """Calculates the Total Average Context Similarity Score for @word.

    Arguments:
    WWC -- Word-Word Cooccurrence Matrix
    fns -- labels for the matrix
    word -- word to calculate the measure for.
    """

    context_vector = WWC[fns.index(word)]
    nonzero_indices = np.flatnonzero(context_vector)

    # The Subset of WWC with just the context vector's rows and columns
    # So that the average can be calculated more efficiently.
    # SWWC = WWC[:,nonzero_indices][nonzero_indices,:]
    SWWC = WWC[nonzero_indices,:]

    # Calculate the cosine distance between each row of SWWC.
    # Gives a Square nxn Matrix with n = number of rows in SWWC
    CSM = distance_matrix(SWWC, metric = metric)

    # Calculates the Average Cosine distance of all pairs of terms.
    # Does NOT count the main diagonal (distance of each row to itself equals 1).
    # That's what the masking is for.
    mask = np.ones(CSM.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    return CSM[mask].mean()

def acds(WWC, fns, word, metric = 'cosine'):
    """Calculates the Average Context Similarity Score of each context term's cooccurrence vector
    to @word's context vector

    """

    context_vector = WWC[fns.index(word)]
    nonzero_indices = np.flatnonzero(context_vector)

    # The Subset of the Cooccurrence Matrix with just the terms that appear in some context.
    SWWC = WWC[nonzero_indices,:]
    # print(SWWC.shape)

    CSM = cdist(SWWC, np.array([context_vector]), metric)
    # print(CSM)
    CSM = CSM
    return CSM.mean()

















