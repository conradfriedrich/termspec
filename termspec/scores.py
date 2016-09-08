# -*- coding: utf-8 -*-


import helpers as util
import matrices

import math
import numpy as np
from scipy.spatial.distance import cdist

def dfs(M, fns, word):
    """Compute the Document Frequency Score of a word from a document-word count matrix.

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
    """Computes the Non Zero Dimensions Score for @word.

    Computes the count of total unique cooccurences for the given word divided by the total of words.
    The result ist the percentage of the words that @word stands in cooccurence with.
    """

    context_vector = M[fns.index(word)]
    n_total_dimensions = len(fns)
    n_non_zero_dimensions = len(context_vector.nonzero()[0])

    return n_non_zero_dimensions / n_total_dimensions

def avnzds (M, fns, word):
    """Computes the Average Context Non Zero Dimensions Score for @word.

    Computes the Nzd Score for every word in the context. Then returns the average.
    """
    context_vector = M[fns.index(word)]

    indices = np.flatnonzero(context_vector)
    indices = indices[indices != 2]

    M = M[indices]

    n_total_dimensions = len(fns)

    def ndzs_per_row( cv ):
        n_non_zero_dimensions = len(cv.nonzero()[0])
        return n_non_zero_dimensions / n_total_dimensions

    nzdss = np.apply_along_axis( ndzs_per_row, axis=1, arr=M )
    return  nzdss.mean()



def tacds(WWC, fns, word, metric = 'cosine'):
    """Computes the Total Average Context Distance Score for @word.

    Arguments:
    WWC -- Word-Word Cooccurrence Matrix
    fns -- labels for the matrix
    word -- word to Compute the measure for.
    """

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)

    # The Subset of WWC with just the context vector's rows
    # So that the average can be Computed more efficiently.
    SWWC = WWC[indices,:]

    # Compute the cosine distance between each row of SWWC.
    # Gives a Square nxn Matrix with n = number of rows in SWWC
    CSM = matrices.distance_matrix(SWWC, metric = metric)

    # Computes the Average Cosine distance of all pairs of terms.
    # Does NOT count the main diagonal (distance of each row to itself equals 1).
    # That's what the masking is for.
    mask = np.ones(CSM.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    return CSM[mask].mean()

def acds(WWC, fns, word, metric = 'cosine'):
    """Computes the Average Context Distance Score of each context term's cooccurrence vector
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

def mdcs(WWC, fns, word, metric = 'cosine', scaled = False):
    """Computes the Mean Context Distance from the Centroid of the Context."""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]
    context_vector = context_vector[indices]
    # The Subset of WWC with just the context vector's rows
    # So that the average can be Computed more efficiently.
    SWWC = WWC[indices,:]
    centroid = np.mean(SWWC, axis=0)

    if metric == 'seuclidean':
        # Variance of the Columns.
        # Mean of every Column
        MEAN = np.mean(SWWC, axis = 0)
        # Square Root of the Standard Deviation
        RSD = SWWC - MEAN
        # Standard Deviations
        SD = RSD*RSD
        # Variance is the mean of the standard deviations
        VARIANCE = np.mean(SD, axis = 0)
        # Can't divide by 0 in all-zero-dimension cases, so just set them to 1
        VARIANCE[VARIANCE == 0] = 1
        # distance to centroid matrix
        DTC = cdist(SWWC, np.array([centroid]), metric, V = VARIANCE)
    else:
        # distance to centroid matrix
        DTC = cdist(SWWC, np.array([centroid]), metric)
    if scaled:
        DTC = DTC * context_vector[:, np.newaxis]

    return DTC.mean()

# DEPRECATED
def depr_mdcs_mc(WWC, fns, word, mc = 50, metric = 'cosine'):
    """Computes the Mean Context Distance from the Centroid of the Context.

    Uses only the @mc most significant co-occurrences!
    """

    context_vector = WWC[fns.index(word)]

    WHOLESUBSETWWC = WWC[np.flatnonzero(context_vector),:]

    # To Account for removal of focus word context vector
    indices = util.mc_indices(context_vector, fns, mc)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be Computed more efficiently.

    # rns = [fns[i] for i in indices]
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

def depr_mdcs_occ(WWC, fns, word, occ = 10, metric = 'cosine'):
    """Computes the Mean Context Distance from the Centroid of the Context."""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be Computed more efficiently.
    SWWC = WWC[indices,:]


    centroid = np.mean(SWWC, axis=0)

    # distance to centroid matrix
    DTC = cdist(SWWC, np.array([centroid]), metric)

    # Return the mean distance to the centroid times the logarithm of occurrence
    return DTC.mean() * math.log(occ)


def depr_sca_mdcs(WWC, fns, word, metric = 'cosine'):
    """Computes the Mean Context Distance from the Centroid of the Context."""

    context_vector = WWC[fns.index(word)]
    indices = np.flatnonzero(context_vector)
    indices = indices[indices != fns.index(word)]

    # The Subset of WWC with just the context vector's rows
    # So that the average can be Computed more efficiently.
    SWWC = WWC[indices,:]

    #Scale the Vectors by Significance of Cooccurrence with Focus word!
    context_vector = context_vector[indices]
    SCALEDSWWC = SWWC * context_vector[:, np.newaxis]

    centroid = np.mean(SCALEDSWWC, axis=0)

    if metric =='seuclidean':
            # Variance of the Columns.
        V = np.mean(SCALEDSWWC, axis = 0)
        # Can't divide by 0 in all-zero-dimension cases, so just set them to 1
        V[V == 0] = 1

        # distance to centroid matrix
        DTC = cdist(SCALEDSWWC, np.array([centroid]), metric, V = V)
    else:
        # distance to centroid matrix
        DTC = cdist(SCALEDSWWC, np.array([centroid]), metric)

    # Return the mean distance to the centroid
    return DTC.mean()