import json
from collections import defaultdict
import numpy as np
from datasets import load_dataset
import unidecode

data = json.load(open("benchmark_llm_summarization/likert_evaluation_results.json"))
criteria = ["coherence","relevance","faithfulness"]

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

cnndm = defaultdict(list)
xsum = defaultdict(list)

for dat in data:
    dataset = cnndm if dat["dataset"] == "cnndm" else xsum
    dataset[dat["article"]].append( dat )

cnnsystems = [x["model"] for x in cnndm[list(cnndm.keys())[0]]]
cnnsystems = [cnnsystems[i] for i in range(0, len(cnnsystems), 3)]
              
xsumsystems = [x["model"] for x in xsum[list(xsum.keys())[0]]]
xsumsystems = [xsumsystems[i] for i in range(0, len(xsumsystems), 3)]

# treating some edge cases
llmid2cnnid = dict()
for i, k in enumerate(cnndm.keys()):
    # for the doc keep iterating over the starting string until we find a single match

    if k not in doc2id:
        if i == 85:
            for kk in doc2id:
                if "Wingers Kevin Mirallas" in kk:
                    matching = kk, doc2id[kk]
                    break
        else:
            prefix_len = 1
            matching = [(x,v) for x,v in doc2id.items() if x.startswith(k[:prefix_len])]
            while len(matching) > 1 and prefix_len < len(k):
                prefix_len +=1
                matching = [(x,v) for x,v in doc2id.items() if x.startswith(k[:prefix_len])]
            assert len(matching) == 1, "{} {} {} {}".format(i,k, len(matching), prefix_len)
            matching = matching[0]
    else:
        matching = k, doc2id[k]
    llmid2cnnid[k] = matching

# cnn
data_out = []

for doc,v in cnndm.items():

    cnndm_doc, (cnndm_id, cnndm_ref) = llmid2cnnid[doc]

    # collate labels
    labels = {k:defaultdict(list) for k in criteria}
    summs = dict()
    for x in v:
        for crit in criteria:
            labels[crit][x["model"]].append(x[crit])
        summs[x["model"]] = x["summary"]
    labels_binary = {x: 1 if sum(v) > 1 else 0 for x,v in labels["faithfulness"].items()}
    labels = {k: {x:np.mean(v) for x,v in labels[k].items()} for k in criteria}
    
    for summ_system in cnnsystems:
        k = summ_system
        v = summs[k]
        data_out.append(
            {
                "id": cnndm_id,
                "model": k,
                "document": unidecode.unidecode(cnndm_doc),
                "summary": unidecode.unidecode(v) if type(v) is str else "",
                "reference": cnndm_ref,
                "coherence": labels["coherence"][k],
                "relevance": labels["relevance"][k],
                "faithfulness": labels["faithfulness"][k],
                "binary_label": labels_binary[k],
            }
        )

# xsum
dataset = load_dataset("xsum")

id2dat = dict()
doc2id = dict()
ref2id = dict()

for i in range(len(dataset["test"])):
    dat = dataset["test"][i]

    id = dat["id"]
    doc = dat["document"].strip().replace("\n", " ").strip()
    summ = dat["summary"].strip().replace("\n", " ").strip()

    id2dat[id] = (doc, summ)
    doc2id[doc] = (id, summ)
    ref2id[summ] = (id, doc)

llmid2xsumid = dict()
for i, k in enumerate(xsum.keys()):
    # for the doc keep iterating over the starting string until we find a single match

    if k not in doc2id:
        if i == -1:
            continue
        else:
            prefix_len = 1
            matching = [(x,v) for x,v in doc2id.items() if x.startswith(k[:prefix_len])]
            while len(matching) > 1 and prefix_len < len(k):
                prefix_len +=1
                matching = [(x,v) for x,v in doc2id.items() if x.startswith(k[:prefix_len])]
            assert len(matching) == 1, "{} {} {} {}".format(i,k, len(matching), prefix_len)
            matching = matching[0]
    else:
        matching = k, doc2id[k]
    llmid2xsumid[k] = matching

# xsum
criteria = ["coherence", "relevance", "faithfulness"]
for doc,v in xsum.items():
    # collate labels
    labels = {k:defaultdict(list) for k in criteria}
    
    xsum_doc, (xsum_id, xsum_ref) = llmid2xsumid[doc]
    
    summs = dict()
    for x in v:
        for crit in criteria:
            labels[crit][x["model"]].append(x[crit])
        summs[x["model"]] = x["summary"]
    labels_binary = {x: 1 if sum(v) > 1 else 0 for x,v in labels["faithfulness"].items()}
    labels = {k: {x:np.mean(v) for x,v in labels[k].items()} for k in criteria}
    for summ_system in xsumsystems:
        k = summ_system
        v = summs[k]
        data_out.append(
            {
                "id": xsum_id,
                "model": k,
                "document": unidecode.unidecode(xsum_doc),
                "summary": unidecode.unidecode(v) if type(v) is str else "",
                "reference": xsum_ref,
                "coherence": labels["coherence"][k],
                "relevance": labels["relevance"][k],
                "faithfulness": labels["faithfulness"][k],
                "binary_label": labels_binary[k],
            }
        )

with open("llmsummeval.jsonl", "w") as f:
    for dat in data_out:
        f.write(json.dumps(dat) + "\n")