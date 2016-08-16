Term Specificity
=================

Small research project testing out a new word space model based method to determine term specificity.

Term Specificity linguistic property of a given word referring to its (lack of) generality. It is sometimes also called Semantic Specificity.

Sparck Jones (1972) proposed the document frequency (as in term frequency inverse document frequency) as a measurement of term specificity. Joho and Sanderson (2007) tested that method on large corpora with some success.

I want to propose two new methods to measure term specificity and evaluate them on terms that are known to have different specificities, e.g. hyperonyms in a hierarchical chain (Joho and Sanderson, 2007).

If viable, those methods are then used to test the hypothesis that german anglicism, words borrowed from the english language, are more specific than their english counterparts. Here additional problems arise for my model, as the corpora used for German and English will not be aligned. But herein also lies the strength of the model, to be able to compare words trough different word space model representations (vector spaces).