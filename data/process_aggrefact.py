"""
Code adapted from AggreFact.
"""
import pandas as pd
import json
import unidecode

import sklearn
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets

df = pd.read_csv("AggreFact/data/aggre_fact_final.csv")

SOTA = ['BART', 'PegasusDynamic', 'T5', 'Pegasus']
XFORMER = ['BertSum', 'BertExtAbs', 'BertExt', 'GPT2', 'BERTS2S', 'TranS2S']
OLD = ['FastAbsRl','TConvS2S', 'PtGen', 'PtGenCoverage',
            'Summa', 'BottomUp', 'Seq2Seq', 'TextRank', 'missing', 'ImproveAbs', 'NEUSUM',
            'ClosedBookDecoder', 'RNES', 'BanditSum', 'ROUGESal', 'MultiTask', 'UnifiedExtAbs']
MAPPING = {0: "SOTA", 1: "XFORMER", 2: "OLD"}

# split data
df_val = df[df.cut == 'val']
df_val_sota = df_val[df_val.model_name.isin(SOTA)]
df_test = df[df.cut == 'test']
df_test_sota = df_test[df_test.model_name.isin(SOTA)]

df_val.to_csv("aggrefact_val.csv", index=False)
df_test.to_csv("aggrefact_test.csv", index=False)

# create data for our case
val_data = []
for i, row in df_val.iterrows():
    val_data.append({
        'id': i, 
        'document': unidecode.unidecode(row['doc'].replace("\n"," ").strip()),
        'summary': unidecode.unidecode(row['summary'].replace("\n", " ").strip()),
    })
with open("aggrefact_val.jsonl", "w") as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

test_data = []
for i, row in df_test.iterrows():
    test_data.append({
        'id': i, 
        'document': unidecode.unidecode(row['doc'].replace("\n"," ").strip()),
        'summary': unidecode.unidecode(row['summary'].replace("\n", " ").strip()),
    })
with open("aggrefact_test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

