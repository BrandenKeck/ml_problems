import os, re, random, torch
import numpy as np
from torch.utils.data import Dataset

# Dictionary Class
class MovieReviewDictionary():
    
    def __init__(self, pad=3000):
        self.pad = pad
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
            tokens[idx] = self.word2idx[word.lower()]
        if len(tokens) >= self.pad: tokens = tokens[:pad]
        else: tokens = np.pad(tokens, (0, self.pad-len(tokens)), 'constant', constant_values=(0,0))
        return tokens.astype('int32')

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
        self.tokens = torch.tensor(np.array(tokens)).to('cuda')
        self.labels = torch.tensor(np.array(labels)).to('cuda')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        label = self.labels[idx]
        return token, label

# Get Batch of Data
def get_batch(size, set="train", pad=3000):

    # Batch variables
    I, X, Y = [], [], []
    hsize = int(np.ceil(size/2))
    
    # Get half positive examples
    pos = os.listdir(f"./{set}/pos")
    pos = random.sample(pos, hsize)
    for pp in pos:
        with open(f'./{set}/pos/{pp}', encoding="utf8") as f:
            lines = f.readlines()[0]
        lines = re.findall(r"[\w']+|[.,!?;]", lines)
        I.append(pp)
        X.append(lines)
        Y.append([0, 1])

    # Get half negative examples
    neg = os.listdir(f"./{set}/neg")
    neg = random.sample(neg, hsize)
    for nn in neg:
        with open(f'./{set}/neg/{nn}', encoding="utf8") as f:
            lines = f.readlines()[0]
        lines = re.findall(r"[\w']+|[.,!?;]", lines)
        I.append(nn)
        X.append(lines)
        Y.append([1, 0])
    
    # Tokenize data
    X = [rdict.tokenize(xx) for xx in X]
    return I, X, Y

# Get a Batch with Size of All Data
def get_all(set="train"):
    return get_batch(25000, set)

def get_dataset(size=10000, set="train"):
    I, X, Y = get_batch(size, set)
    return MovieReviewDataset(X, Y)
