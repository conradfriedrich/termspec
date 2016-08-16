# -*- coding: utf-8 -*-

import nltk
import codecs
import sys
import spinfoutils as utils

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize

en_corpus = PlaintextCorpusReader('/Users/conrad/Documents/uni/2016SS/SpinfoHausarbeit/py/en',['en_wiki_cool.txt'])
de_corpus = PlaintextCorpusReader('/Users/conrad/Documents/uni/2016SS/SpinfoHausarbeit/py/de',['de_wiki_cool.txt'])


def coocc_simple(sents):
    # Counts every occurrence of a term from the given sentence
    # baiscally just a frequence
    coolcooccurrences = {}
    for coolsent in sents:
        for term in coolsent:
            if term in coolcooccurrences:
                coolcooccurrences[term] += 1
            else:
                coolcooccurrences[term] = 1
    return coolcooccurrences

# for fileid in en_corpus.fileids():
#     sents = en_corpus.sents(fileid)
#     sents = [normalize(sent) for sent in sents]
#     # print(sents)
#     coolsents = [sent for sent in sents if 'cool' in sent]
#     print(len(coolsents))

#     coolcooccurrences = coocc_simple(coolsents)
#     for w in sorted(coolcooccurrences, key=coolcooccurrences.get, reverse=True)[:50]:
#         if not w == 'cool':
#             print (w, coolcooccurrences[w])
#     print(len(coolcooccurrences))

for fileid in de_corpus.fileids():
    sents = de_corpus.sents(fileid)
    sents = [utils.normalize(sent, 'german') for sent in sents]
    coolsents = [sent for sent in sents if 'cool' in sent]
    coolcooccurrences = coocc_simple(coolsents)

    
    for w in sorted(coolcooccurrences, key=coolcooccurrences.get, reverse=True)[:30]:
        print (w, coolcooccurrences[w])


# # teststr = "U.S.A.! U.S.A:! There is no single concept of cool. Here is a random number: 12876 715 091283-122 9192! One of the essential characteristics of cool is its mutability—what is considered cool changes over time and varies among cultures and generations."
# teststr = "Der Begriff wird einerseits zur saloppen Bezeichnung einer besonders gelassenen oder lässigen, nonchalanten, kühlen, souveränen, kontrollierten und nicht nervösen Geisteshaltung oder Stimmung genutzt (vergleiche: Kühl bleiben, kühler Kopf im Sinne von „ruhig bleiben“). [1] Andererseits ist cool als jugendsprachliches Wort zur Kennzeichnung von als besonders positiv empfundenen, den Idealvorstellungen entsprechenden Sachverhalten (ähnlich wie „geil“) gebräuchlich im Sinne von „schön“, „gut“, „angenehm“ oder „erfreulich“.[2] Zudem ist das Wort – je nach Milieu und Altersstufe – extrem vielseitig einsetzbar."
# # sents = nltk.sent_tokenize(teststr)
# sent_tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
# sents = sent_tokenizer.tokenize(teststr)

# for sent in sents:
#     words = word_tokenize(sent)
#     print(words)
#     words = normalize(words,'german')
#     print (words)

# for sent in sents2:
#     words = word_tokenize(sent)
#     print (words)


