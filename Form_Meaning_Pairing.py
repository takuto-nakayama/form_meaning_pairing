from elmoformanylangs import Embedder
from sklearn.cluster import DBSCAN
from transformers import AutoTokenizer

import numpy as np
import os, math, re, itertools, statistics

class FMP:
    def __init__(self, lang:str,  min_freq:int):
        self.lang = lang  #the name of the language
        self.min = min_freq  #the minimum frequency of tokens
        self.sentences = [l.strip() for l in open(f'sentences/{self.lang}.txt', encoding='utf-8').readlines()]  #a list of sentences
        self.elmo = Embedder(f'models/{self.lang}/')  #model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2')  #tokenizer
        os.mkdir(f'{self.lang}')  #create a directory that contains the result
        
    def tokenize(self):  #tokenize into words
        self.sentences = [self.tokenizer.tokenize(s) for s in self.sentences]
        for s in self.sentences:
            while ' ##' in ' '.join(s):
                for i, tkn in enumerate(s):
                    if re.match('##', tkn):
                        s[i-1] = s[i-1]+re.sub('##', '', tkn)
                        del s[i]

    def contextualize(self):  #pick up sentences that include the word in question
        os.mkdir(f'{self.lang}/contexts')
        list_tokens = list(itertools.chain.from_iterable(self.sentences))
        set_tokens = set(list_tokens)
        token_indice = {}
        for i, t in enumerate(set_tokens):
            if list_tokens.count(t) >= self.min:
                idx_sentences = []
                with open(f'{self.lang}/contexts/{t}_{i}.txt', 'w') as f:
                    for j, s in enumerate(self.sentences):
                        if t in s:
                            idx_sentences.append(j)
                            f.write(' '.join(s) + '\n')
                token_indice[t] = idx_sentences
        return token_indice

    def embed(self, token_indice):  #get vectors of the words considering their contexts
        os.mkdir(f'{self.lang}/embeddings')
        list_embeddings = []
        embeddings = self.elmo.sents2elmo(self.sentences)
        for tkn in token_indice.keys():
            embed_t = []
            for idx_s in token_indice[tkn]:
                for idx_t in [i for i, x in enumerate(self.sentences[idx_s]) if x == tkn]:
                    embed_t.append(list(embeddings[idx_s][idx_t]))
            list_embeddings.append(embed_t)
            np.savetxt(f'{self.lang}/embeddings/{tkn}_{idx_t+idx_s}.csv', np.array(embed_t), delimiter=',')
        return list_embeddings
    
    def estimate(self, embeddings, epsilon, min_samples):  #estimate the number of meanings of the words
        list_prob = []
        for embed in embeddings:
            np_dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit_predict(embed)
            list_prob.append([np.sum(np_dbscan == c)/len(np_dbscan) for c in range(max(np_dbscan))])
        return list_prob

    def entropy(self, list_prob):  #calculate the entropy of the language
        list_entropy = []
        for prob in list_prob:
            list_entropy.append(sum([-p * math.log(p, 2) for p in prob]))
        return statistics.mean(list_entropy)
