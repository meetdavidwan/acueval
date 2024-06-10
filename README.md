# ACUEval:Fine-grained Hallucination Evaluation and Correction for Abstractive Summarization (ACL 2024)

This repository contains the code for the paper "ACUEval: Fine-grained Hallucination Evaluation and Correction for Abstractive Summarization"

**Authors:** [David Wan](https://meetdavidwan.github.io), [Koustuv Sinha](https://koustuvsinha.com), [Srinivasan Iyer](https://sriniiyer.github.io), [Asli Celikyilmaz](http://asli.us), [Mohit Bansal](https://www.cs.unc.edu/~mbansal), and [Ramakanth Pasunuru](http://www.rama-kanth.com)

**Arxiv:** SOON!

## 1. Generating Summaries with Lookahead

### 1.1. Environment
Needed packages:
- PyTorch
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- pandas
- numpy
- unidecode
- scikit-learn
- scipy

### 1.2 Dataset
- [SummEval](https://github.com/Yale-LILY/SummEval)
- [LLMSummEval](https://github.com/Tiiiger/benchmark_llm_summarization)
- [AggreFact](https://github.com/Liyan06/AggreFact)

Please clone the repositories under `data` so that you have the following folders: `data/AggreFact, data/benchmark_llm_summarization, data/SummEval`.

To process the files, run the respective processing code:
```
cd data
python process_aggrefact.py
python process_summeval.py
python process_llmsummeval.py
cd ..
```

### 1.3 ACUEval metric
The code is under `src`. We provide the two steps separately (`acu_generation.py` and `acu_verification.py`), as well as a combined version (`acueval.py`).

All three files takes in two parameters, the input jsonl file and the output jsonl file. `acu_generation.py` adds `acus` entry to each json, which is a list of strings.`acu_verification.py` adds `acu_predictions` (list of floats) and `acueval_score` (float).

Example:
```
# run the two steps separately
python src/acu_generation.py data/summeval.jsonl data/summeval_with_acus.jsonl
python src/acu_verification.py data/summeval_with_acus.jsonl data/summeval_with_acus_and_predictions.jsonl

# alternatively, run everything together
python src/acueval.py data/summeval.jsonl data/summeval_with_acus_and_predictions.jsonl
```

The LM's code is wrapped in `src/model.py`, which hardcodes StableBeluga. If you would like to try other models or have different models for the two steps, feel free to extend the class.

### 1.4 Evaluation

`src/compute_bacc.py` computes the balanced accuracy. The code takes in the prediction jsonl file and uses `binary_label` and `acuveval_score` entry.


## Citation

```
@inproceedings{wan-etal-2024-acueval,
    title = "ACUEval:Fine-grained Hallucination Evaluation and Correction for Abstractive Summarization",
    author = "Wan, David  and
      Sinha, Koustuv  and
      Iyer, Srinivasan and
      Celikyilmaz, Asli and
      Bansal, Mohit and
      Pasunuru, Ramakanth",
      booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
      year={2024}
}
```