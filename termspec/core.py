# -*- coding: utf-8 -*-

import helpers as util
import numpy as np
import math

from timer import Timer

from nltk.corpus import brown
from nltk import word_tokenize, sent_tokenize, collocations

from scipy.spatial.distance import pdist, cdist, squareform
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

word_pairs = [
    ('food','beverage'),
    ('food','dessert'),
    ('food','bread'),
    ('food','cheese'),
    ('food','meat'),
    ('food','dish'),
    ('food','butter'),
    ('food','cake'),
    ('food','egg'),
    ('food','candy'),
    ('food','pastry'),
    ('food','vegetable'),
    ('food','fruit'),
    ('food','sandwich'),
    ('food','soup'),
    ('food','pizza'),
    ('food','salad'),
    ('food', 'relish'),
    ('food', 'olives'),
    ('food', 'ketchup'),
    ('food', 'cookie'),

    ('beverage', 'alcohol'),
    # ('beverage', 'cola'),

    ('alcohol','liquor'),
    ('alcohol','gin'),
    ('alcohol','rum'),
    ('alcohol','brandy'),
    ('alcohol','cognac'),
    ('alcohol','wine'),
    ('alcohol','champagne'),

    ('meat', 'liver'),
    ('meat', 'ham'),

    ('dish', 'sandwich'),
    ('dish','soup'),
    ('dish','pizza'),
    ('dish','salad'),

    ('vegetable', 'tomato'),
    ('vegetable', 'mushroom'),
    ('vegetable', 'legume'),

    ('fruit', 'pineapple'),
    ('fruit', 'apple'),
    ('fruit', 'peaches'),
    ('fruit', 'strawberry'),

    ('vehicle','truck'),
    ('vehicle','car'),
    ('vehicle','trailer'),
    ('vehicle','campers'),

    ('car', 'jeep'),
    ('car','cab'),
    ('car','coupe'),

    ('person','worker'),
    ('person','writer'),
    ('person','intellectual'),
    ('person','professional'),
    ('person','leader'),
    ('person','entertainer'),
    ('person','engineer'),

    ('worker','editor'),
    ('worker','technician'),
    ('writer','journalist'),
    ('writer','commentator'),
    ('writer','novelist'),

    ('intellectual','physicist'),
    ('intellectual','historian'),
    ('intellectual','chemist'),

    ('professional','physician'),
    # ('professional','educator'),
    ('professional','nurse'),
    ('professional','dentist'),

    ('entity','organism'),
    ('entity','object'),

    ('animal', 'mammal'),
    ('animal', 'bird'),
    ('animal','dog'),
    ('animal','cat'),
    ('animal','horse'),
    ('animal','chicken'),
    ('animal','duck'),
    ('animal','fish'),
    ('animal','turtle'),
    ('animal','snake'),

    ('mammal','cattle'),
    ('mammal','dog'),
    ('mammal','cat'),
    ('mammal','horse'),

    ('bird', 'chicken'),
    ('bird', 'duck'),

    ('fish', 'herring'),
    ('fish', 'salmon'),
    ('fish', 'trout'),

    ('metal', 'alloy'),
    ('metal', 'steel'),
    ('metal', 'gold'),
    ('metal', 'silver'),
    ('metal', 'iron'),

    ('location', 'region'),
    ('location', 'country'),
    ('location', 'state'),
    ('location', 'city'),

    ('substance', 'food'),
    ('substance', 'metal'),
    ('substance', 'carcinogen'),
    ('substance', 'fluid'),
    ('fluid', 'water'),
    ('commodity', 'clothing'),
    ('commodity', 'appliance'),
    ('artifact', 'covering'),
    ('artifact', 'paint'),
    ('artifact', 'roof'),
    ('artifact', 'curtain'),
    ('publication', 'book'),
    ('publication', 'article'),
    ('artifact', 'decoration'),
    ('artifact', 'drug'),
    ('fabric', 'nylon'),
    ('fabric', 'wool'),
    ('fabric', 'cotton'),
    ('facility', 'airport'),
    ('facility', 'headquarters'),
    ('facility', 'station'),
    ('structure', 'house'),
    ('structure', 'factory'),
    ('structure', 'store'),
    ('organ', 'heart'),
    ('organ', 'lung')
    ]

def easy_setup_sentence_context(filename = None, corpus = 'toy', deserialize = True, serialize = True):
    """Sets up data object for experiments with the WORD CONTEXT = SENTENCES

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
            data['docs'] = util.read_from_file(filename + '_docs' + '.tmp')
            data['DWF'] = util.read_from_file(filename + '_DWF' + '.tmp')

            # The large matrizes have been serialized in sparse format.
            WWC_S = util.read_from_file(filename + '_WWC' + '.tmp')
            WWDICE_S = util.read_from_file(filename + '_WWDICE' + '.tmp')
            data['WWC'] = np.asarray(WWC_S.todense())
            data['WWDICE'] = np.asarray(WWDICE_S.todense())
            data['fns'] = util.read_from_file(filename + '_fns' + '.tmp')

        print ('##### Deserialized Data in %4.1fs' % t.secs)

    # If any one of these fails, recompute all.
    if not (data and data['docs'] and data['DWF'] and data['WWC'] and data['WWDICE'] and data['fns']):
        data = {}
        with Timer() as t:
            docs = retrieve_data_and_tokenize_sentences(corpus = corpus)

            # Put the tokens all in one large array to throw out low frequency words
            tokens = []
            for doc in docs:
                for sent in doc:
                    tokens.extend(sent)

            print('tokens in corpus', len(tokens))
            words = util.frequency_threshold(tokens, fqt = 10)

            print('words', len(words))
            words = set(words)

            # Restructure tokens as documents. Quite costly.
            data['docs'] = []
            count = 0
            for doc in docs:
                return_doc = []
                for sent in doc:
                    return_sent = [token for token in sent if token in words]
                    # Append if not empty.
                    if return_sent:
                        return_doc.append(return_sent)
                        count += len(return_sent)
                if return_doc:
                    data['docs'].append(return_doc)
            print ('tokens after reducing', count)

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
        print('Feature Names (Words): ', len(fns))
        if filename and serialize:
            with Timer() as t:
                util.write_to_file(filename + '_docs' + '.tmp', data['docs'])
                util.write_to_file(filename + '_DWF' + '.tmp', data['DWF'])
                util.write_to_file(filename + '_fns' + '.tmp', data['fns'])

                # Let's save about 99% of disk space.
                WWC_S = sparse.csr_matrix(data['WWC'])
                WWDICE_S = sparse.csr_matrix(data['WWDICE'])

                util.write_to_file(filename + '_WWC' + '.tmp', WWC_S)
                util.write_to_file(filename + '_WWDICE' + '.tmp', WWDICE_S)
            print ('##### Serialized Data in %4.1fs' % t.secs)
        print()
    return data

def easy_setup_context_window(
    fqt = 10,
    window_size = 4,
    score_fn = 'dice',
    filename = None,
    corpus = 'toy',
    deserialize = True,
    serialize = True):

    """Sets up data object for experiments with the WORD CONTEXT = CONTEXT WINDOW
    """

    def raw_count(*marginals):
        """Scores ngrams by their frequency"""
        return marginals[0]

    tokens = words = WWC = None
    f_prx = filename
    f_sfx = '.tmp'
    tokens_filename = f_prx + '_tokens' + '_' + corpus + '_ws' + str(window_size) + '_' + score_fn + f_sfx
    words_filename = f_prx + '_words' + '_' + corpus + '_ws' + str(window_size) + '_' + score_fn + f_sfx
    WWC_filename = f_prx + '_wwmatrix' + '_' + corpus + '_ws' + str(window_size) + '_' + score_fn + f_sfx

    if filename and deserialize:


        tokens = util.read_from_file(tokens_filename)
        words = util.read_from_file(words_filename)
        WWC = util.read_from_file(WWC_filename)

    if not (tokens and words and WWC):
        with Timer() as t:
            tokens = retrieve_data_and_tokenize_tokens(corpus = corpus)
        print ('##### Retrieved data and Tokenized in %4.1fs' % t.secs)

        print('totaltokens', len(tokens))

        # Reduce the tokens to words that appear more often than fqt.

        words = util.frequency_threshold(tokens, fqt)
        words = set(words)
        tokens = [token for token in tokens if token in words]

        print('totalwords', len(words))
        print('totaltokens after substraction', len(tokens))

        bgm    = collocations.BigramAssocMeasures()
        finder = collocations.BigramCollocationFinder.from_words(tokens, window_size = window_size)

        if score_fn == 'dice':
            scored = finder.score_ngrams( bgm.chi_sq )
        elif score_fn == 'phi_sq':
            scored = finder.score_ngrams( bgm.phi_sq )
        elif score_fn == 'chi_sq':
            scored = finder.score_ngrams( bgm.chi_sq )
        elif score_fn == 'raw_freq':
            scored = finder.score_ngrams( bgm.raw_freq )
        elif score_fn == 'raw_count':
            scored = finder.score_ngrams( raw_count )
        else:
            raise ValueError('Passed Score Function not implemented', score_fn)

        # For faster Reference, Create a dictionary with words as keys and indices as values.
        # Not very memory efficient, but does enhance performance.
        words = list(words)
        words.sort()
        words_indices = {}
        for i, word in enumerate(words):
            words_indices[word] = i

        with Timer() as t:
            # Create Word-Word-Cooccurrence Matrix
            WWC = np.zeros( (len(words), len(words)) )
            print(WWC.shape)
            for collocation in scored:
                pair = collocation[0]
                WWC[words_indices[pair[0]], words_indices[pair[1]]] = collocation[1]
        print('##### Created Word Word Matrix in %4.1fs' % t.secs)

        if filename and serialize:
            with Timer() as t:

                util.write_to_file(tokens_filename, tokens)
                util.write_to_file(words_filename, words)
                util.write_to_file(WWC_filename, WWC)
            print ('##### Serialized Data in %4.1fs' % t.secs)

    data = { 'tokens': tokens, 'words': words, 'WWC': WWC }
    return data


def retrieve_data_and_tokenize_tokens(corpus = 'toy'):
    """Retrieves the data from the corpus and makes a neat array of tokens.
    [ 'doc1_token1', 'doc1_token2, ..., 'doc_token_m' ]
    """

    if corpus == 'toy':
        docs = [
            'Optional plotz says to frobnicate the bizbaz first. foo bar bar. foo',
            'Return a foo bang. foo bar. The Ding Dangel Dong Dong foo bang bar.',
            'foo bang baz padauts remoreng.',
            'foo bar bar baz foo'
        ]

        tokens = []
        for doc in docs:
            doc = sent_tokenize(doc)
            for sent in doc:
                sent = word_tokenize(sent)
                sent = util.normalize(sent)
                tokens.extend(sent)


    elif corpus == 'brown':
        # tokens = brown.words(categories = ['news'])
        tokens = brown.words()
        tokens = util.normalize(tokens)

    elif corpus == 'reuters':
        tokens = brown.words()
        tokens = util.normalize(tokens)

    elif corpus == 'brown_reuters':
        tokens = brown.words()
        tokens = util.normalize(tokens)
        tokens2 = corpus.reuters.words()
        tokens2 = util.normalize(tokens2)
        tokens.extend(tokens2)
    else:
        raise ValueError('Corpus passed is not implemented.', corpus)

    return tokens


def retrieve_data_and_tokenize_sentences(corpus = 'toy'):
    """Retrieves the data from the corpus and makes a neat array of tokens, sentences and documents out of it.
    [ #docs
        [ #document1
            ['token1','token2','token3'], #sentence1
            ['token1','token2','token3'], #sentence2
        ],
        [ #document2
            ['token1','token2','token3'], #sentence1
            ['token1','token2','token3'], #sentence2
        ]
    ]

    """

    return_docs = []

    # Compute from Sample Sentences.
    if corpus == 'toy':

        docs = [
            'Optional plotz says to frobnicate the bizbaz first. foo bar bar. foo',
            'Return a foo bang. foo bar. The Ding Dangel Dong Dong foo bang bar.',
            'foo bang baz padauts remoreng.',
            'foo bar bar baz foo'
        ]

        return_docs = []
        for doc in docs:
            words = word_tokenize(doc)
            words = util.normalize(words)
            return_docs.append(words)

    # Compute from brown corpus.
    elif corpus == 'brown':

        categories = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

        docs = [brown.sents(fileids = fileid) for fileid in brown.fileids(categories=categories)]

        ##################################

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
    """Calculates the Total Average Context Distance Score for @word.

    Arguments:
    WWC -- Word-Word Cooccurrence Matrix
    fns -- labels for the matrix
    word -- word to calculate the measure for.
    """

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.
    SWWC = WWC[indices,:]

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
    """Calculates the Average Context Distance Score of each context term's cooccurrence vector
    to @word's context vector

    """

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)

    # The Subset of the Cooccurrence Matrix with just the terms that appear in some context.
    SWWC = WWC[indices,:]
    # print(SWWC.shape)

    CSM = cdist(SWWC, np.array([context_vector]), metric)
    # print(CSM)
    return CSM.mean()

def mdcs(WWC, fns, word, metric = 'cosine'):
    """Calculates the Mean Context Distance from the Centroid of the Context."""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.
    SWWC = WWC[indices,:]


    centroid = np.mean(SWWC, axis=0)

    # distance to centroid matrix
    DTC = cdist(SWWC, np.array([centroid]), metric)

    # Return the mean distance to the centroid
    return DTC.mean()

def mdcs_occ(WWC, fns, word, occ = 10, metric = 'cosine'):
    """Calculates the Mean Context Distance from the Centroid of the Context."""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.
    SWWC = WWC[indices,:]


    centroid = np.mean(SWWC, axis=0)

    # distance to centroid matrix
    DTC = cdist(SWWC, np.array([centroid]), metric)

    # Return the mean distance to the centroid times the logarithm of occurrence
    return DTC.mean() * math.log(occ)

def mdcs_mc(WWC, fns, word, mc = 50, metric = 'cosine'):
    """Calculates the Mean Context Distance from the Centroid of the Context.

    Uses only the @mc most significant co-occurrences!
    """

    context_vector = WWC[fns.index(word)]

    WHOLESUBSETWWC = WWC[np.flatnonzero(context_vector),:]

    # To Account for removal of focus word context vector
    indices = util.mc_indices(context_vector, fns, mc)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.

    rns = [fns[i] for i in indices]
    # print(rns)
    # print()
    SWWC = WWC[indices,:]

    # util.printprettymatrix(SWWC, cns = fns, rns = rns)
    # print()
    # SWWC = WWC[np.argsort(context_vector)[::-1],:]
    centroid = np.mean(WHOLESUBSETWWC, axis=0)

    # distance to centroid matrix
    DTC = cdist(SWWC, np.array([centroid]), metric)
    # util.printprettymatrix(DTC, rns = rns)

    # Return the mean distance to the centroid
    return DTC.mean()


def mdcs_sca(WWC, fns, word, metric = 'cosine'):
    """Calculates the Mean Context Distance from the Centroid of the Context."""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.
    SWWC = WWC[indices,:]

    #Scale the Vectors by Significance of Cooccurrence with Focus word!
    context_vector = context_vector[indices]
    SCALEDSWWC = SWWC * context_vector[:, np.newaxis]

    centroid = np.mean(SCALEDSWWC, axis=0)

    # distance to centroid matrix
    DTC = cdist(SCALEDSWWC, np.array([centroid]), metric)

    # Return the mean distance to the centroid
    return DTC.mean()

## Todo: Compute Variance of Cluster
def se_mdcs(WWC, fns, word):
    """Calculates the Standard Euclidean Distance using Variance"""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.
    SWWC = WWC[indices,:]
    centroid = np.mean(SWWC, axis=0)

    # Variance of the Columns.
    V = np.mean(SWWC, axis = 0)
    # Can't divide by 0 in all-zero-dimension cases, so just set them to 1
    V[V == 0] = 1

    # distance to centroid matrix
    DTC = cdist(SWWC, np.array([centroid]), metric = 'seuclidean', V = V)

    # Return the mean distance to the centroid
    return DTC.mean()

def se_mdcs_mc(WWC, fns, word, mc = 50):
    """Calculates the Standard Euclidean Distance using Variance"""

    context_vector = WWC[fns.index(word)]

    # If the context vector has more nonzero elements than mc, only take the mc occurrences!
    indices = util.mc_indices(context_vector, fns, mc)

    #removes the focus terms's own vector from the context
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.
    SWWC = WWC[indices,:]
    centroid = np.mean(SWWC, axis=0)

    # Variance of the Columns.
    V = np.mean(SWWC, axis = 0)
    # Can't divide by 0 in all-zero-dimension cases, so just set them to 1
    V[V == 0] = 1

    # distance to centroid matrix
    DTC = cdist(SWWC, np.array([centroid]), metric = 'seuclidean', V = V)

    # Return the mean distance to the centroid
    return DTC.mean()


## Todo: Compute Variance of Cluster
def se_mdcs_sca(WWC, fns, word):
    """Calculates the Standard Euclidean Distance using Variance"""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be calculated more efficiently.
    SWWC = WWC[indices,:]
    context_vector = context_vector[indices]
    SCALEDSWWC = SWWC * context_vector[:, np.newaxis]


    centroid = np.mean(SCALEDSWWC, axis=0)

    # Variance of the Columns.
    V = np.mean(SCALEDSWWC, axis = 0)
    # Can't divide by 0 in all-zero-dimension cases, so just set them to 1
    V[V == 0] = 1

    # distance to centroid matrix
    DTC = cdist(SCALEDSWWC, np.array([centroid]), metric = 'seuclidean', V = V)

    # Return the mean distance to the centroid
    return DTC.mean()