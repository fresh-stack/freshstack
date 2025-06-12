<div align="center">
  <h1>FreshStack</h1>
  <p>A Repository for Constructing Realistic IR/RAG Benchmarks</p>
</div>

<p align="center"><img width=500 src="https://raw.githubusercontent.com/fresh-stack/freshstack/main/images/freshstack-logo-cropped.png"/></p>

<h4 align="center">
    <p>
        <a href="https://arxiv.org/abs/2504.13128">Paper</a> |
        <a href="https://fresh-stack.github.io/">Website</a> |
        <a href="https://fresh-stack.github.io/#leaderboard">Leaderboard</a> |
        <a href="https://huggingface.co/freshstack">Dataset</a>
    <p>
</h4>

**FreshStack** is a modular framework to **automatically build realistic IR/RAG benchmarks** from niche, community-sourced technical content (e.g., Stack Overflow + GitHub repositories). It supports:

* Scraping **human-asked queries** based on StackOverflow.
* Gathering **up-to-date corpora** via chunking any GitHub repository.
* **Retrieval evaluation** of any dense/multi-vector model on the Freshstack repository.
* Datasets released under **CC-BY-SA 4.0** and code and scripts under **Apache 2.0 License**.

## Installation

Install via pip, tested with Python 3.10+:

```python
pip install freshstack
```

If you want to build from source, use:

```bash
git clone https://github.com/fresh-stack/freshstack.git
cd freshstack
pip install -e .
```

## 🚀 Quickstart: Load Freshstack Dataset
```python
from freshstack.datasets import DataLoader

freshstack_dataloader = DataLoader(
    queries_repo="freshstack/queries-oct-2024", 
    corpus_repo="freshstack/corpus-oct-2024",
    topic="langchain") # or "yolo", "angular", "laravel" or "godot"

# Loads the corpus, queries and nuggets in the BEIR format
corpus, queries, nuggets = dataloader.load(split="test")

# Loads the qrels (nuggets), qrels (query) and query to nugget mapping
qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split="test")
```

## 🚀 Quickstart: Model Evaluation

### 1. Evaluate only the retrieved results
```python
# Your runfile can be stored as a .txt in the following format: [qid, Q0, docid, 0, score, run_name], e.g.,
# 76185522 Q0 angular/adev/src/content/tutorials/learn-angular/steps/14-routerLink/answer/src/app/app.component.ts_0_368 0 0.7353782057762146 your_model_name

from freshstack import util
from freshstack.retrieval.evaluation import EvaluateRetrieval

# retrieval_results: dict[str, dict[str, str]] with qid: {doc_id: score}
retrieval_results = util.load_runfile("<path_to_your_runfile>")
evaluator = EvaluateRetrieval(k_values=[10, 20, 50])
alpha_ndcg, coverage, recall = evaluator.evaluate(
    qrels_nuggets=qrels_nuggets,
    query_to_nuggets=query_to_nuggets,
    qrels_query=qrels_query,
    results=retrieval_results,
)
```

### 2. Evaluate any dense embedding model (e.g., Qwen3-0.6B-embedding) using BEIR.
> Make sure you install the latest PyPI BEIR repository: `pip install beir`

```python
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval as BEIREval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from freshstack.retrieval.evaluation import EvaluateRetrieval

# Custom query prompt for evaluating the Qwen3-0.6B model on Freshstack.
query_prompt = "Instruct: Given a technical question, retrieve relevant code snippets or technical documentation that best answer the question\nQuery: "

model = DRES(models.SentenceBERT(
    "Qwen/Qwen3-Embedding-0.6B",
    max_length=2048, # IMP: keep the max_length for both query & passage atleast 2048 tokens.
    prompts={"query": query_prompt, "passage": ""},
    model_kwargs={
        "attn_implementation": "flash_attention_2", 
        "device_map": "auto", 
        "torch_dtype": "bfloat16"
    },
    tokenizer_kwargs={"padding_side": "left"},
), batch_size=32)

retriever = BEIREval(model, score_function="cos_sim")
retrieval_results = retriever.retrieve(corpus=corpus, queries=queries)

# Evaluate and compute retrieval score once you have results
evaluator = EvaluateRetrieval(k_values=[10, 20, 50])
alpha_ndcg, coverage, recall = evaluator.evaluate(
    qrels_nuggets=qrels_nuggets,
    query_to_nuggets=query_to_nuggets,
    qrels_query=qrels_query,
    results=retrieval_results,
)
```

### 3. Evaluate any multi-vector model (e.g., ColBERT) using Pylate.
> Make sure you install the latest PyLate repository: `pip install pylate`.

```python
from pylate import indexes, models, retrieve
from freshstack.retrieval.evaluation import EvaluateRetrieval

# Step 1: Load the ColBERT model
model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
    query_length=2048, document_length=2048
)

# Step 2: Initialize the Voyager index or (PLAID index)
index = indexes.Voyager(
    index_folder=f"./langchain_index",
    index_name="index",
    override=False,  # This overwrites the existing index if any
)

# Step 3: Encode the documents and add them to index
documents_ids = list(corpus.keys())
documents_embeddings = model.encode(
    [doc["text"] for doc in corpus.values()],
    batch_size=32,
    is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
)

index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)

# Step 5: Compute query embeddings
query_ids = list(queries.keys())
queries_embeddings = model.encode(
    list(queries.values()),
    batch_size=32,
    is_query=True,  # Ensure that it is set to False to indicate that these are queries
)

# Step 6: Initialize the ColBERT retriever with the Voyager index & retrieve documents
retriever = retrieve.ColBERT(index=index)
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=50,  # Retrieve top-k results based on the maximum k value specified
    batch_size=1,  # We have kept a batch size of 1 to avoid memory issues.
    device="cpu",  # Use CPU for inference, change to "cuda" if you have a GPU available.
)

# Step 7: Prepare the results in the required BEIR format
retrieval_results = {}
for query_id, doc_scores in zip(query_ids, scores):
    retrieval_results[query_id] = {}
    for doc_id, score in doc_scores:
        retrieval_results[query_id][doc_id] = score

# Step 8: Evaluate and compute retrieval score once you have results
evaluator = EvaluateRetrieval(k_values=[10, 20, 50])
alpha_ndcg, coverage, recall = evaluator.evaluate(
    qrels_nuggets=qrels_nuggets,
    query_to_nuggets=query_to_nuggets,
    qrels_query=qrels_query,
    results=retrieval_results,
)
```
---

## 📚 Raw Freshstack Datasets (Oct 2024)

The raw freshstack datasets can be downloaded via HF:

* langchain, yolo, laravel, angular, godot (via [huggingface.co](https://huggingface.co/datasets/freshstack/queries-oct-2024))

```python
from datasets import load_dataset
queries = load_dataset("freshstack/queries-oct-2024", subset="yolo")
corpus = load_dataset("freshstack/corpus-oct-2024", subset="yolo")
```

---

## 🧭 Project Structure

```
freshstack/
├─ examples/            # contains examples
│   ├─ chunking/        # examples for github repo chunking
│   ├─ evaluation/      # examples for model eval on freshstack
├─ freshstack/          # core logic modules
│   ├─ retrieval/       # code for retrieval evaluation
│   ├─ datasets/        # code for freshstakc dataloader
│   └─ chunking/        # code for github repo chunking
└─ pyproject.toml
```

---
## FreshStack Leaderboard

The upto date leaderboard for Freshstack (version oct-2024) is provided here: [https://fresh-stack.github.io/#leaderboard](https://fresh-stack.github.io/#leaderboard). 

> NOTE: Below is the snapshot of the Freshstack leaderboard from Jun 12th 2025.

| Model Name                           | Size | Date       | AVERAGE α@10 | AVERAGE C\@20 | AVERAGE R\@50 | LANGCHAIN α@10 | LANGCHAIN C\@20 | LANGCHAIN R\@50 | YOLO α@10 | YOLO C\@20 | YOLO R\@50 | LARAVEL α@10 | LARAVEL C\@20 | LARAVEL R\@50 | ANGULAR α@10 | ANGULAR C\@20 | ANGULAR R\@50 | GODOT α@10 | GODOT C\@20 | GODOT R\@50 |
| ------------------------------------ | ---- | ---------- | ------------ | ------------- | ------------- | -------------- | --------------- | --------------- | --------- | ---------- | ---------- | ------------ | ------------- | ------------- | ------------ | ------------- | ------------- | ---------- | ----------- | ----------- |
| Oracle: Fusion (BM25; ...) (Nuggets) | -    | 2024‑11‑01 | 0.541        | 0.868         | 0.755         | 0.519          | 0.881           | 0.655           | 0.601     | 0.876      | 0.825      | 0.566        | 0.888         | 0.818         | 0.544        | 0.881         | 0.756         | 0.476      | 0.815       | 0.719       |
| Oracle: BM25 (Nuggets)               | -    | 2024‑11‑01 | 0.488        | 0.768         | 0.556         | 0.467          | 0.739           | 0.446           | 0.519     | 0.796      | 0.657      | 0.540        | 0.840         | 0.654         | 0.485        | 0.787         | 0.536         | 0.428      | 0.680       | 0.489       |
| Oracle: Voyage Large 2 (Nuggets)     | -    | 2024‑11‑01 | 0.404        | 0.769         | 0.586         | 0.419          | 0.763           | 0.508           | 0.430     | 0.845      | 0.675      | 0.409        | 0.791         | 0.624         | 0.406        | 0.733         | 0.533         | 0.353      | 0.715       | 0.590       |
| Oracle: BGE (Gemma-2) (Nuggets)      | 9B   | 2024‑11‑01 | 0.389        | 0.735         | 0.547         | 0.308          | 0.667           | 0.405           | 0.461     | 0.784      | 0.572      | 0.448        | 0.806         | 0.666         | 0.393        | 0.755         | 0.536         | 0.335      | 0.664       | 0.555       |
| Qwen3‑8B (Emb)                    | 8B   | 2025‑06‑05 | 0.365        | 0.689         | 0.525         | 0.331          | 0.694           | 0.423           | 0.393     | 0.728      | 0.567      | 0.421        | 0.748         | 0.615         | 0.373        | 0.700         | 0.502         | 0.307      | 0.576       | 0.521       |
| Qwen3‑4B (Emb)                    | 4B   | 2025‑06‑05 | 0.347        | 0.656         | 0.490         | 0.320          | 0.675           | 0.415           | 0.404     | 0.744      | 0.550      | 0.402        | 0.748         | 0.604         | 0.304        | 0.618         | 0.442         | 0.303      | 0.496       | 0.440       |
| Fusion (BM25; BGE; E5; Voyage)       | -    | 2024‑11‑01 | 0.343        | 0.669         | 0.539         | 0.337          | 0.700           | 0.477           | 0.304     | 0.627      | 0.534      | 0.425        | 0.748         | 0.646         | 0.385        | 0.719         | 0.532         | 0.265      | 0.550       | 0.505       |
| Oracle: E5 (Mistral-7B) (Nuggets)    | 7B   | 2024‑11‑01 | 0.337        | 0.664         | 0.496         | 0.323          | 0.684           | 0.432           | 0.437     | 0.737      | 0.554      | 0.287        | 0.631         | 0.532         | 0.346        | 0.670         | 0.470         | 0.292      | 0.596       | 0.494       |
| Stella‑1.5B v5                       | 1.5B | 2025‑01‑01 | 0.317        | 0.615         | 0.479         | 0.315          | 0.660           | 0.388           | 0.334     | 0.624      | 0.559      | 0.370        | 0.681         | 0.590         | 0.330        | 0.630         | 0.414         | 0.237      | 0.481       | 0.443       |
| Voyage Large 2                       | -    | 2024‑11‑01 | 0.289        | 0.589         | 0.438         | 0.246          | 0.528           | 0.308           | 0.270     | 0.570      | 0.453      | 0.345        | 0.701         | 0.543         | 0.304        | 0.625         | 0.427         | 0.282      | 0.522       | 0.458       |
| Stella‑400M v5                       | 400M | 2025‑01‑01 | 0.276        | 0.578         | 0.422         | 0.285          | 0.608           | 0.356           | 0.241     | 0.538      | 0.447      | 0.320        | 0.648         | 0.534         | 0.288        | 0.619         | 0.359         | 0.244      | 0.476       | 0.412       |
| BGE (Gemma-2)                        | 9B   | 2024‑11‑01 | 0.269        | 0.569         | 0.427         | 0.216          | 0.548           | 0.337           | 0.258     | 0.547      | 0.430      | 0.348        | 0.699         | 0.574         | 0.323        | 0.571         | 0.378         | 0.200      | 0.479       | 0.419       |
| Qwen3‑0.6B (Emb)                  | 596M | 2025‑06‑05 | 0.262        | 0.543         | 0.394         | 0.259          | 0.588           | 0.369           | 0.260     | 0.504      | 0.383      | 0.288        | 0.593         | 0.463         | 0.253        | 0.535         | 0.356         | 0.249      | 0.495       | 0.400       |
| E5 (Mistral-7B)                      | 7B   | 2024‑11‑01 | 0.255        | 0.553         | 0.397         | 0.304          | 0.654           | 0.393           | 0.243     | 0.552      | 0.394      | 0.250        | 0.565         | 0.470         | 0.262        | 0.548         | 0.368         | 0.217      | 0.444       | 0.359       |
| GTE (large) v1.5                     | 434M | 2024‑01‑09 | 0.226        | 0.494         | 0.318         | 0.206          | 0.470           | 0.252           | 0.195     | 0.445      | 0.271      | 0.318        | 0.626         | 0.482         | 0.284        | 0.578         | 0.343         | 0.127      | 0.348       | 0.240       |
| BM25                                 | -    | 2024‑11‑01 | 0.218        | 0.448         | 0.316         | 0.230          | 0.475           | 0.261           | 0.137     | 0.342      | 0.337      | 0.319        | 0.602         | 0.441         | 0.259        | 0.551         | 0.340         | 0.144      | 0.268       | 0.200       |
| Nomic Embed (Code)                | 7B   | 2025‑03‑24 | 0.218        | 0.488         | 0.348         | 0.224          | 0.518           | 0.292           | 0.227     | 0.539      | 0.390      | 0.222        | 0.532         | 0.407         | 0.237        | 0.511         | 0.356         | 0.178      | 0.341       | 0.295       |
| CodeRankEmbed                        | 137M | 2024‑11‑03 | 0.104        | 0.279         | 0.162         | 0.099          | 0.271           | 0.128           | 0.075     | 0.215      | 0.128      | 0.108        | 0.324         | 0.225         | 0.146        | 0.363         | 0.167         | 0.091      | 0.224       | 0.160       |


### 👥 Contribute your model to the leaderboard

1. Fork the repo (https://github.com/fresh-stack/fresh-stack.github.io)
2. Create a new branch (`git checkout -b <your_branch>`)
3. Add your model scores in the following format to the `leaderboard_data.json`:
```bash
{
    "leaderboardData": [
        {
            "info": {
                "name": "BM25",
                "size": "-",
                "type": "open_source",
                "date": "2024-11-01",
                "link": "https://github.com/castorini/pyserini"
            },
            "datasets": {
                "langchain": {"alpha_ndcg_10": 0.230, "coverage_20": 0.475, "recall_50": 0.261},
                "yolo":      {"alpha_ndcg_10": 0.137, "coverage_20": 0.342, "recall_50": 0.337},
                "laravel":   {"alpha_ndcg_10": 0.319, "coverage_20": 0.602, "recall_50": 0.441},
                "angular":   {"alpha_ndcg_10": 0.259, "coverage_20": 0.551, "recall_50": 0.340},
                "godot":     {"alpha_ndcg_10": 0.144, "coverage_20": 0.268, "recall_50": 0.200},
                "average":   {"alpha_ndcg_10": 0.218, "coverage_20": 0.448, "recall_50": 0.316},
            }
        },
        ...
    ]
}
```

4. Submit a pull request, ideally including:

   * The updated `leaderboard_data.json`
   * Pipeline invocation script (reference)
   * Brief evaluation summary (reference)

All contributions welcome—especially new domain expansions, evaluation improvements, and retrieval baselines!

---

## 📄 Citation

If you use FreshStack in your work, please cite:

```bib
@article{thakur-freshstack:2025,
  author       = {Nandan Thakur and
                  Jimmy Lin and
                  Sam Havens and
                  Michael Carbin and
                  Omar Khattab and
                  Andrew Drozdov},
  title        = {FreshStack: Building Realistic Benchmarks for Evaluating Retrieval
                  on Technical Documents},
  journal      = {CoRR},
  volume       = {abs/2504.13128},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2504.13128},
  doi          = {10.48550/ARXIV.2504.13128},
  eprinttype    = {arXiv},
  eprint       = {2504.13128},
  timestamp    = {Thu, 22 May 2025 21:00:35 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2504-13128.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
The main contributors of this repository are:
- [Nandan Thakur](https://github.com/thakur-nandan), Personal Website: [thakur-nandan.gitub.io](https://thakur-nandan.github.io)

Contact person: Nandan Thakur, [nandant@gmail.com](mailto:nandant@gmail.com)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

---

## Collaboration

This project is developed in collaboration with the following organizations:

| Organization           | Logo                                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| University of Waterloo | <img src="https://raw.githubusercontent.com/fresh-stack/freshstack/main/images/uwaterloo.png" alt="University of Waterloo logo" /> |
| Databricks             | <img src="https://raw.githubusercontent.com/fresh-stack/freshstack/main/images/databricks-logo.png" alt="Databricks logo" />     |
