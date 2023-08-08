import os, re, random, torch
import numpy as np

# Dictionary Class
class ReviewDictionary():
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.build_dictionary()
    
    def tokenize(self, words):
        tokens = np.zeros(len(words))
        for idx, word in enumerate(words):
            if word.lower() not in self.word2idx:
                self.add_word(word.lower())
            tokens[idx] = self.word2idx[word.lower()]
        return tokens

    def wordize(self, tokens):
        words = np.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            words[idx] = self.idx2word[token]
        return words

    def build_dictionary(self):
        with open(f'./vocab.txt') as f:
            for line in f.readlines():
                word = line.rstrip()
                self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)

rdict = ReviewDictionary() # Make Global


def get_batch(size, set="train"):

    pos = os.listdir(f"./{set}/pos")
    neg = os.listdir(f"./{set}/neg")

    X, Y = [], []
    hsize = int(np.ceil(size/2))
    pos = random.sample(pos, hsize)
    neg = random.sample(neg, hsize)

    for pp in pos:
        with open(f'./{set}/pos/{pp}') as f:
            lines = f.readlines()[0]
        lines = re.findall(r"[\w']+|[.,!?;]", lines)
        X.append(lines)
        Y.append(1)

    for nn in neg:
        with open(f'./{set}/neg/{nn}') as f:
            lines = f.readlines()[0]
        lines = re.findall(r"[\w']+|[.,!?;]", lines)
        X.append(lines)
        Y.append(0)
    
    X = [rdict.tokenize(xx) for xx in X]
    return X, Y

