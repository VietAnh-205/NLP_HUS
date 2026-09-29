import math 
from collections import Counter 
import numpy as np 

class NGramLanguegeModel: 

    def __init__(self, n): 
        self.n = n 
        self.vocabulary = set() 
        self.unigram_counts = Counter() 
        self.bigram_counts = Counter() 
        self.trigram_counts = Counter() 

        self.total_tokens = 0 

    def build_vocabulary(self, corpus): 
        self.vocabulary = set() 
        for sentence in corpus: 
            for word in sentence: 
                self.vocabulary.add(word) 
        return self.vocabulary

    def count_ngrams(self, corpus): 
        self.unigram_counts = Counter() 
        self.bigram_counts = Counter() 
        self.trigram_counts = Counter() 

        for sentence in corpus: 
            for word in sentence: 
                self.unigram_counts[word] += 1 

            for i in range(len(sentence) - 1): 
                bigram = (sentence[i], sentence[i + 1]) 
                self.bigram_counts[bigram] += 1 

            for i in range(len(sentence) - 2): 
                trigram = (sentence[i], sentence[i + 1], sentence[i + 2]) 
                self.trigram_counts[trigram] += 1

        self.total_tokens = sum(self.unigram_counts.values())

    def train_unigram(self): 
        self.unigram_probs = {} 
        for word, count in self.unigram_counts.items(): 
            self.unigram_probs[word] = (count / self.total_tokens) 

    def train_bigram(self): 
        self.bigram_probs = {} 
        for (w1, w2), counts in self.bigram_counts.items(): 
            self.bigram_probs[(w1, w2)] = (counts / self.unigram_counts[w1]) 

    def train_trigram(self): 
        self.trigram_probs = {} 
        for (w1, w2, w3), count in self.trigram_counts.item(): 
            self.trigram_probs[(w1, w2, w3)] = (count / self.bigram_counts[(w1, w2)]) 

    def fit(self, corpus): 
        self.build_vocabulary(corpus) 
        self.count_ngrams(corpus) 
        self.train_unigram() 

        if self.n >= 2: 
            self.train_bigram() 

        if self.n >= 3: 
            self.train_trigram() 

        return self 

    def probability(self, context, word): 
        if self.n == 1: 
            return self.unigram_probs.get(word, 0) 
        elif self.n == 2: 
            if len(context) != 1: 
                raise ValueError(
                    "Bigram context must contain 1 word"
                )
            w1 = context[0] 
            return self.bigram_probs.get((w1, word), 0)
        elif self.n == 3: 
            if len(context) != 2: 
                raise ValueError(
                    "Trigram context must contain 2 word"
                )
            w1, w2 = context
            return self.trigram_probs.get((w1, w2, word), 0) 
    
    def sentence_probability(self, sentence): 
        probability = 1.0 
        if self.n == 1: 
            for word in sentence: 
                p = self.probability(
                    (), 
                    word
                )
                probability *= p 
        elif self.n == 2: 
            probability *= self.probability(
                (), 
                sentence[0] 
            )
            for i in range(1, len(sentence)): 
                context = (sentence[i - 1], ) 
                p = self.probability(context, sentence[i]) 
                probability *= p 
        elif self.n == 3: 
            probability *= self.probability(
                (), 
                sentence[0]
            )

            if len(sentence) >= 2: 
                probability *= self.probability(
                    (sentence[0], ), 
                    sentence[1]
                )
            for i in range(2, len(sentence)): 
                context = (
                    sentence[i -2],
                    sentence[i -1]
                )
                p = self.probability(
                    context, 
                    sentence[i] 
                )

                probability *= p 

        return probability

#log probability 
    def sentence_log_probability(self, sentence): 
        probability = 1.0 
        if self.n == 1: 
            for word in sentence: 
                p = self.probability(
                    (), 
                    word
                )
                probability *= np.log(p)
        elif self.n == 2: 
            probability *= self.probability(
                (), 
                sentence[0] 
            )
            for i in range(1, len(sentence)): 
                context = (sentence[i - 1], ) 
                p = self.probability(context, sentence[i]) 
                probability *= np.log(p)
        elif self.n == 3: 
            probability *= self.probability(
                (), 
                sentence[0]
            )

            if len(sentence) >= 2: 
                probability *= self.probability(
                    (sentence[0], ), 
                    sentence[1]
                )
            for i in range(2, len(sentence)): 
                context = (
                    sentence[i -2],
                    sentence[i -1]
                )
                p = self.probability(
                    context, 
                    sentence[i] 
                )

                probability *= np.log(p)

        return probability


    def next_word_distribution(self, context): 
        distribution = {} 
        for word in self.vocabulary: 
            p = self.probability(context, word) 

        if p > 0 : 
            distribution[word] = p 

        return dict(
            sorted(
                distribution.item(), 
                key = lambda x: x[1], 
                reverse = True
            )
        )


# Test 
corpus = [
    ["I", "love", "NLP"], 
    ["I", "love", "AI"], 
    ["I", 'study', 'NLP']
]

model = NGramLanguegeModel(2) 
model.fit(corpus) 
print(model.vocabulary) 
print(model.unigram_counts) 
