from datasets import load_dataset
import json
from collections import defaultdict, Counter
import numpy as np
import unidecode

annotations = [json.loads(line) for line in open("SummEval/model_annotations.aligned.scored.jsonl")]

dataset = load_dataset("cnn_dailymail","3.0.0")

id2dat = dict()
doc2id = dict()
ref2id = dict()

for i in range(len(dataset["test"])):
    dat = dataset["test"][i]

    id = dat["id"]
    doc = dat["article"].strip().replace("\n", " ").strip()
    summ = dat["highlights"].strip().replace("\n", " ").strip()

    id2dat[id] = (doc, summ)
    doc2id[doc] = (id, summ)
    ref2id[summ] = (id, doc)

documents = []
summaries = []
binary_labels = []

for ann in annotations:
    dat = id2dat[ann["id"].split("-")[-1]]
    documents.append(dat[0].strip().replace("\n", " "))
    summaries.append(ann["decoded"])

    labels = [lab["consistency"] for lab in ann["expert_annotations"]]
    c = Counter(labels)
    lab = c.most_common()[0][0]
    binary_labels.append( 1 if lab == 5 else 0)

data = []
for i, (doc, summ, lab) in enumerate(zip(documents, summaries, binary_labels)):
    data.append(
        {
            "id": i,
            "document": unidecode.unidecode(doc),
            "summary": unidecode.unidecode(summ),
            "binary_label": lab,
        }
    )

# skip the 17th system
data = [d for i,d in enumerate(data) if i%17 !=0]

with open("summeval.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")