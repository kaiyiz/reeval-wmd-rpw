import os
import argparse
import time
from tqdm import tqdm

import scipy.io
import numpy as np
from sklearn.metrics import pairwise_distances
import pickle

from cot import rpw, transport_lmr
import ot

from util import generate_data_name

def make_input_valid4lmr(a, b, D):
    # make len(a) = len(b), by add 1e-6s to the shorter one, and also make D a square matrix with 1e-6s
    # C.shape[0] = len(SB), C.shape[1] = len(SA)
    if len(a) < len(b):
        a = np.concatenate((a, 1e-6 * np.ones(len(b) - len(a))))
    elif len(a) > len(b):
        b = np.concatenate((b, 1e-6 * np.ones(len(a) - len(b))))
    if D.shape[0] < D.shape[1]:
        D = np.concatenate((D, 1e-6 * np.ones((D.shape[1] - D.shape[0], D.shape[1]))))
    elif D.shape[0] > D.shape[1]:
        D = np.concatenate((D, 1e-6 * np.ones((D.shape[0], D.shape[0] - D.shape[1]))), axis=1)
    # normalize a and b
    a = a / a.sum()
    b = b / b.sum()
    return a, b, D

def input_checker(DA, SB, C):
    """
    This function checks if the given demand and supply arrays and cost matrix are valid.
    """
    assert np.allclose(np.sum(DA), 1, atol=1e-6), "The sum of the demand array should be equal to 1."
    assert np.allclose(np.sum(SB), 1, atol=1e-6), "The sum of the supply array should be equal to 1."
    if not (len(DA) == len(SB)):
        raise ValueError("The length of the demand and supply arrays should be equal.")
    if C.shape[0] != len(SB) or C.shape[1] != len(DA):
        raise ValueError("The cost matrix should have the same dimensions as the demand and supply arrays.")
    return True

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='bbcsport')
parser.add_argument('--clean', type=bool, default=False)
parser.add_argument('--reduced', type=bool, default=False)
parser.add_argument('--rpw', type=bool, default=False)
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--part', type=int, default=0)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--tfidf', action='store_true', default=False)
parser.add_argument('--delta', type=float, default=0.0003)
parser.add_argument('--k', type=float, default=0.1)
parser.add_argument('--p', type=float, default=1.0)
args = parser.parse_args()

# mainname = args.filename.split('/')[-1]
dtrain = 'train' if args.train else 'none'
dtfidf = 'tfidf' if args.tfidf else 'bow'
delta = args.delta
kk = args.k
p = args.p
is_rpw = args.rpw
# print all arguments together
print(args)

mainname = generate_data_name(args.filename, args.clean, args.reduced, False)
filename = 'data/{}.mat'.format(mainname)
data = scipy.io.loadmat(filename)

if 'X' in data:
    X = np.vstack([x.T for x in data['X'][0]])
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    freq = np.bincount(inverse)
    N = len(data['X'][0])
else:
    X = np.vstack([x.T for x in data['xtr'][0]] + [x.T for x in data['xte'][0] if len(x.T) > 0])
    _, inverse = np.unique(X, axis=0, return_inverse=True)
    freq = np.bincount(inverse)
    N = len(data['xtr'][0]) + len(data['xte'][0])

if 'X' in data:
    leftX = 'X'
    rightX = 'X'
    leftBOW = 'BOW_X'
    rightBOW = 'BOW_X'
    leftind = np.cumsum([0] + [x.shape[1] for x in data['X'][0]])
    rightind = np.cumsum([0] + [x.shape[1] for x in data['X'][0]])
elif dtrain == 'train':
    leftX = 'xtr'
    rightX = 'xtr'
    leftBOW = 'BOW_xtr'
    rightBOW = 'BOW_xtr'
    leftind = np.cumsum([0] + [x.shape[1] for x in data['xtr'][0]])
    rightind = np.cumsum([0] + [x.shape[1] for x in data['xtr'][0]])
else:
    leftX = 'xte'
    rightX = 'xtr'
    leftBOW = 'BOW_xte'
    rightBOW = 'BOW_xtr'
    rightind = np.cumsum([0] + [x.shape[1] for x in data['xtr'][0]])
    leftind = np.cumsum([rightind[-1]] + [x.shape[1] for x in data['xte'][0]])


n = len(data[leftX][0])
m = len(data[rightX][0])
print("Computing distances for {}x{} pairs".format(n, m))

pair_list = [(i, j) for i in range(n) for j in range(m)]

ids = [0]
for i in range(args.split):
    nex = ids[-1] + len(pair_list) // args.split
    if i < len(pair_list) % args.split:
        nex += 1
    ids.append(nex)

assert(ids[-1] == len(pair_list))

start = ids[args.part]
end = ids[args.part+1]

vals = []
total = end - start
t0 = time.time()

print("Computing {} distance for {} pairs".format("RPW" if is_rpw else "EMD", total))
if is_rpw:
    print(f"k: {kk}, delta: {delta}, p: {p}")
    
for k, (i, j) in tqdm(enumerate(pair_list[start:end]), desc="Processing", total=total):
    if data[leftX][0, i].shape[1] == 0 or data[rightX][0, j].shape[1] == 0:
        vals.append((i, j, -1))
        continue
    D = pairwise_distances(data[leftX][0, i].T, data[rightX][0, j].T)
    a = data[leftBOW][0, i][0].astype(float)
    b = data[rightBOW][0, j][0].astype(float)
    if args.tfidf:
        a = a * np.log(N / freq[inverse[leftind[i]:leftind[i+1]]])
        b = b * np.log(N / freq[inverse[rightind[j]:rightind[j+1]]])
    a /= a.sum()
    b /= b.sum()
    # T = ot.emd(a, b, D)
    D = D / D.max()
    a, b, D = make_input_valid4lmr(a, b, D)
    input_checker(a, b, D)
    if is_rpw:
        try: 
            val = rpw(b, a, D, delta, kk, p)
            if np.isnan(val):
                print(f"invalid value at {i}, {j}")
                emd_val = ot.emd2(a, b, D)
                print(f"EMD value: {emd_val}")
                val = emd_val
        except:
            print(f"Error at {i}, {j}, use EMD instead")
            emd_val = ot.emd2(a, b, D)
            val = emd_val
    else:
        val = ot.emd2(a, b, D)

    # val = (T * D).sum()
    vals.append((i, j, val))

if not os.path.exists('out'):
    os.mkdir('out')

mainname = generate_data_name(args.filename, args.clean, args.reduced, is_rpw)
print(f"Saving to out/{mainname}-{start}-{end}-{dtrain}-{dtfidf}.pickle")
with open('out/{}-{}-{}-{}-{}.pickle'.format(mainname, start, end, dtrain, dtfidf), 'wb') as f:
    pickle.dump(vals, f)