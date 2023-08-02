import os, re, random
import numpy as np
test_pos = os.listdir("./test/pos")
test_neg = os.listdir("./test/neg")

def get_batch(size):

    X, Y = [], []
    hsize = np.ceil(size/2)
    pos = random.choice(test_pos, k=hsize)
    neg = random.choice(test_neg, k=hsize)

    for pp in pos:
        with open(f'./test/pos/{pp}') as f:
            lines = f.readlines()
        lines = re.sub('[^0-9a-zA-Z]+', '*', lines)
        X.append(lines)
        Y.append(1)

    for nn in neg:
        with open(f'./test/neg/{nn}') as f:
            lines = f.readlines()
        lines = re.sub('[^0-9a-zA-Z]+', '*', lines)
        X.append(lines)
        Y.append(1)
    
