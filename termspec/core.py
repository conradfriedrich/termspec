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
    tokens_filename = f_prx + '_' + corpus + '_ws' + str(window_size) + '_' + score_fn + '_tokens' + f_sfx
    words_filename = f_prx  + '_' + corpus + '_ws' + str(window_size) + '_' + score_fn + '_words' + f_sfx
    WWC_filename = f_prx + '_' + corpus + '_ws' + str(window_size) + '_' + score_fn + '_wwmatrix' + f_sfx

    if filename and deserialize:

        tokens = util.read_from_file(tokens_filename)
        words = util.read_from_file(words_filename)
        WWC_S = util.read_from_file(WWC_filename)
        if WWC_S:
            WWC = np.asarray(WWC_S.todense())

    if not (tokens and words and WWC.any()):
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
                WWC_S = sparse.csr_matrix(WWC)
                util.write_to_file(WWC_filename, WWC_S)
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