import os, re, random, torch
import numpy as np
from torch.utils.data import Dataset

# Dictionary Class
class MovieReviewDictionary():
    
    def __init__(self):
        self.word2idx = {"NULL":0}
        self.idx2word = ["NULL"]
        self.build_dictionary()
    
    def tokenize(self, words, add=True):
        tokens = np.zeros(len(words))
        for idx, word in enumerate(words):
            if word.lower() not in self.word2idx:
                if add:
                    self.add_word(word.lower())
                else:
                    tokens[idx] = 0
                    continue
            tokens[idx] = int(self.word2idx[word.lower()])
        return tokens

    def wordize(self, tokens):
        words = np.empty(len(tokens), dtype=object)
        for idx, token in enumerate(tokens):
            words[idx] = self.idx2word[int(token)]
        return words

    def build_dictionary(self):
        with open(f'./vocab.txt', encoding="utf8") as f:
            for line in f.readlines():
                word = line.rstrip()
                self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)-1
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)

rdict = MovieReviewDictionary() # Make Global

# Dataset Class
class MovieReviewDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        label = self.labels[idx]
        return token, label

# Get Batch of Data
def get_batch(size, set="train", pad=3000):

    pos = os.listdir(f"./{set}/pos")
    neg = os.listdir(f"./{set}/neg")

    I, X, Y = [], [], []
    hsize = int(np.ceil(size/2))
    pos = random.sample(pos, hsize)
    neg = random.sample(neg, hsize)

    for pp in pos:
        with open(f'./{set}/pos/{pp}', encoding="utf8") as f:
            lines = f.readlines()[0]
        lines = re.findall(r"[\w']+|[.,!?;]", lines)
        I.append(pp)
        X.append(lines)
        Y.append(1)

    for nn in neg:
        with open(f'./{set}/neg/{nn}', encoding="utf8") as f:
            lines = f.readlines()[0]
        lines = re.findall(r"[\w']+|[.,!?;]", lines)
        I.append(nn)
        X.append(lines)
        Y.append(0)
    
    for idx in np.arange(len(X)):
        if len(X[idx]) >= pad:
            X[idx] = X[idx][:pad]
        else:
            X[idx] = np.pad(X[idx], (0,pad-len(X[idx])), 'constant', constant_values=(0,0))

    X = [rdict.tokenize(xx).astype('int32') for xx in X]
    return I, X, Y

# Get a Batch with Size of All Data
def get_all(set="train"):
    return get_batch(25000, set)

I_train, X_train, Y_train = get_all("train")
train_data = MovieReviewDataset(X_train, Y_train)
I_test, X_test, Y_test = get_all("test")
test_data = MovieReviewDataset(X_test, Y_test)

