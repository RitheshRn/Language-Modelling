#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        # Set flag1 = 1 when reuters only
        flag1 = 0
        if flag1 == 1:
            with open('reuters_train_trigrams.txt', 'w') as f:
                for s in corpus:
                    f.write("%s\n" % str(" ".join(s)))
        
        for s in corpus:
            s = [ss.lower() for ss in s]
            self.fit_sentence(s)
            
#         # Add sentences from Reuters
#         with open('reuters_sentences_to_append.txt', 'r') as f:
#             temp1 = f.readlines()
            
        self.vocabsize = len(self.vocab())
        
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 3 # for EOS(1) + SOS(2) 
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(2, len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

class Trigram(LangModel):
    def __init__(self, backoff = 0.000001):
        from collections import defaultdict
        self.model = dict()
        self.model_val = defaultdict(int)
        self.lbackoff = log(backoff, 2)
        
        
    def inc_word(self, w):
        if (w[0], w[1]) in self.model:
            if w[2] in self.model[(w[0], w[1])]:
                self.model[(w[0], w[1])][w[2]] += 1
            else:
                self.model[(w[0], w[1])] = {w[2]: 1}
        else:
            self.model[(w[0], w[1])] = {w[2]: 1}
            
    def inc_word_val(self, w):
        self.model_val[w] += 1
    
    def fit_sentence_val(self, sentence):
        sentence.insert(0,'START_OF_SENTENCE')
        sentence.insert(0,'START_OF_SENTENCE')
        for words in range(len(sentence)-2):
            self.inc_word_val((sentence[words], sentence[words+1], sentence[words+2]))
        if len(sentence)>1:
            self.inc_word_val((sentence[-2], sentence[-1], 'END_OF_SENTENCE'))
    
    def fit_sentence(self, sentence):
        sentence.insert(0,'START_OF_SENTENCE')
        sentence.insert(0,'START_OF_SENTENCE')
        for words in range(len(sentence)-2):
            self.inc_word((sentence[words], sentence[words+1], sentence[words+2]))
        if len(sentence)>1:
            self.inc_word((sentence[-2], sentence[-1], 'END_OF_SENTENCE'))
        
    def norm(self):
        """Normalize and convert to log2-probs."""
        delta = 0.05
        for words12 in self.model:
            total = 0
            for word3 in self.model[words12]:
                total += self.model[words12][word3]
            total += self.vocabsize*delta
            ltotal = log(total, 2)
            for word3 in self.model[words12]:
                self.model[words12][word3] = log(self.model[words12][word3] + (1*delta), 2) - ltotal
    
    
    ### HANDLE unseen sentences 
    def cond_logprob(self, word, previous):
        if len(previous)<2:
            return -log(self.vocabsize, 2)
        
        ### REMEMBER TO CHECK IF self.model[(previous[-2], previous[-1])] EXISTS IN THE FIRST PLACE
        try:
            if word in self.model[(previous[-2], previous[-1])]:
                return self.model[(previous[-2], previous[-1])][word]
            else:
                return -log(self.vocabsize, 2)
        except:
            return -log(self.vocabsize, 2)
            
    def vocab(self):
        keep = set()
        for w12 in self.model:
            keep.add(w12[0])
            keep.add(w12[1])
            for w3 in self.model[w12]:
                keep.add(w3)
        
        return list(keep)
    
    def frequent_trigrams(self, corpus):
        """
        get frequent trigrams from the valiation set
        """
        for s in corpus:
            s = [ss.lower() for ss in s]
            self.fit_sentence_val(s)
            
        temp = sorted(self.model_val.items(), key=lambda x:x[1], reverse=True)
        with open('reuters_frequent_trigrams.txt', 'w') as f:
            for item in temp:
                f.write("%s\n" % str(" ".join(item[0])))
        
        
   