"""
Code adpated from Aggrefact's evaluation.
"""
from datasets import load_dataset
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import math
import pandas as pd

from sklearn.metrics import balanced_accuracy_score

from collections import defaultdict
import random
random.seed(42)

def choose_best_threshold(labels, scores):
    best_f1 = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 0.2)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        f1_score = balanced_accuracy_score(labels, preds)

        if f1_score >= best_f1:
            best_f1 = f1_score
            best_thresh = thresh
    return best_thresh, best_f1

def resample_balanced_acc(preds, labels, n_samples=1000, sample_ratio=0.8):
    N = len(preds)
    idxs = list(range(N))
    N_batch = int(sample_ratio*N)

    bal_accs = []
    for _ in range(n_samples):
        random.shuffle(idxs)
        batch_preds = [preds[i] for i in idxs[:N_batch]]
        batch_labels = [labels[i] for i in idxs[:N_batch]]
        
        bal_accs.append(balanced_accuracy_score(batch_labels, batch_preds))
    return bal_accs

data = [json.loads(line) for line in open(sys.argv[1])]

labels = [dat["binary_label"] for dat in data]
preds = np.array([dat["acuveval_score"] for dat in data])

# # SummEval: remove 17th system
# preds = [p for i,p in enumerate(preds) if i%17 != 0]
# labels = [l for i,l in enumerate(labels) if i%17 !=0]

preds = [p if p is not None else 0. for p in preds]

preds_val, preds_test, labels_val, labels_test = [], [], [], []
for i, (p,l) in enumerate(zip(preds, labels)):
    if i %2 == 0:
        preds_val.append(p)
        labels_val.append(l)
    else:
        preds_test.append(p)
        labels_test.append(l)

best_t, best_f = choose_best_threshold(labels_val, preds_val)
print("validation threshold",best_t, best_f)

preds_test_ = [1 if p > best_t else 0 for p in preds_test]
# print(balanced_accuracy_score(labels_test, preds_test_))

P5 = 5 / 2 # Correction due to the fact that we are running 2 tests with the same data
P1 = 1 / 2 # Correction due to the fact that we are running 2 tests with the same data

sampled_batch_preds = []
    
samples = resample_balanced_acc(preds_test_, labels_test)
sampled_batch_preds.append(samples)
low5, high5 = np.percentile(samples, P5), np.percentile(samples, 100-P5)
low1, high1 = np.percentile(samples, P1), np.percentile(samples, 100-P1)
bacc = balanced_accuracy_score(labels_test, preds_test_)

print(" - %.3f, %.3f" % (bacc, bacc-low5))