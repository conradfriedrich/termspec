Term Specificity
=================

Small research project testing out a new word space model based method to determine term specificity.

Term specificity is a linguistic property of a word referring to its (lack of) generality. It is sometimes also called semantic specificity.

Sparck Jones (1972) proposed the document frequency (as in term frequency inverse document frequency) as a measurement of term specificity. Joho and Sanderson (2007) tested that method on large corpora with some success.

I want to propose two new scores to measure term specificity and evaluate them with terms that are known to have different specificities, e.g. hyperonyms in a hierarchical chain (Joho and Sanderson, 2007).

If viable, those scores are then used to test the hypothesis that german anglicisms, words borrowed from the english language, are more specific than their english counterparts. Here additional problems arise, as the corpora used for German and English will not be aligned. That means that the words to compare do not share the same word space model, they lie in different vector spaces. The strength of the new scores is that they should be able to compare the term specificity of words over different word space model representations (vector spaces).