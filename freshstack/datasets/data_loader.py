from __future__ import annotations

import logging
from collections import defaultdict

from datasets import Value, load_dataset

from .util import nuggets_to_query_qrels

logger = logging.getLogger(__name__)

Topics = [
    "oct-2024",
]


class DataLoader:
    def __init__(
        self,
        queries_repo: str = "freshstack/queries-oct-2024",
        corpus_repo: str = "freshstack/corpus-oct-2024",
        topic: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
    ):
        self.corpus = {}
        self.queries = {}
        self.nuggets = {}
        self.answers = {}

        # Qrels and Nuggets
        self.qrels_answers = defaultdict(dict)
        self.qrels_nuggets = defaultdict(dict)
        self.qrels_query = defaultdict(dict)
        self.query_nuggets = defaultdict(dict)

        self.hf_queries_repo = queries_repo
        self.hf_corpus_repo = corpus_repo
        self.subset = topic
        self.keep_in_memory = keep_in_memory
        self.streaming = streaming

    def load(self, split="test") -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, str]]:
        if not len(self.corpus):
            logger.info(f"Loading {self.subset} Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.items())[0])

        if not len(self.queries):
            logger.info(f"Loading {self.subset} Queries...")
            self._load_queries(split)
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.items())[0])

        self._load_nuggets_and_qrels(split)
        logger.info(f"Loading {self.subset} Nuggets and Qrels...")
        logger.info("Loaded %d %s Nuggets.", len(self.nuggets), split.upper())
        logger.info("Nugget Example: %s", list(self.nuggets.items())[0])

        return self.corpus, self.queries, self.nuggets

    def load_qrels(
        self, split="test"
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, list[str]]]:
        if not len(self.qrels_nuggets):
            logger.info(f"Loading {self.subset} Qrels...")
            self._load_nuggets_and_qrels(split)
            logger.info("Loaded %d %s Qrels.", len(self.qrels), split.upper())
            logger.info("Qrel Example: %s", list(self.qrels.items())[0])

        # Convert nuggets to query qrels
        self.query_to_nuggets, self.qrels_query = nuggets_to_query_qrels(self.qrels_nuggets)
        return self.qrels_nuggets, self.qrels_query, self.query_to_nuggets

    def load_answers(self, split="test") -> dict[str, str]:
        if not len(self.answers):
            logger.info(f"Loading {self.subset} Answers...")
            self._load_answers(split)
            logger.info("Loaded %d %s Answers.", len(self.answers), split.upper())
            logger.info("Answer Example: %s", self.answers[0])

        return self.answers

    def load_corpus(self) -> dict[str, dict[str, str]]:
        if not self.hf_repo:
            self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        if self.hf_corpus_repo:
            corpus_ds = load_dataset(
                self.hf_corpus_repo,
                self.subset,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=True,
            )["train"]
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [col for col in corpus_ds.column_names if col not in ["id", "text", "title"]]
        )
        # convert corpus_ds to a dictionary with id as key
        self.corpus = {row["id"]: {"text": row["text"], "title": row.get("title", "")} for row in corpus_ds}

    def _load_queries(self, split):
        if self.hf_queries_repo:
            queries_ds = load_dataset(
                self.hf_queries_repo,
                self.subset,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=True,
            )[split]
        queries_ds = queries_ds.cast_column("query_id", Value("string"))
        queries_ds = queries_ds.rename_column("query_id", "id")
        queries_ds = queries_ds.rename_column("query_text", "text")
        queries_ds = queries_ds.rename_column("query_title", "title")
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text", "title"]]
        )
        # convert corpus_ds to a dictionary with id as key
        self.queries = {row["id"]: row.get("title", "") + " " + row["text"] for row in queries_ds}

    def _load_answers(self, split):
        if self.hf_queries_repo:
            answer_ds = load_dataset(
                self.hf_queries_repo,
                self.subset,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=True,
            )[split]
        answer_ds = answer_ds.cast_column("query_id", Value("string"))
        answer_ds = answer_ds.rename_column("answer_text", "text")
        answer_ds = answer_ds.remove_columns(
            [col for col in answer_ds.column_names if col not in ["query_id", "text"]]
        )
        # convert corpus_ds to a dictionary with id as key
        self.answers = {row["id"]: row["text"] for row in answer_ds}

    def _load_nuggets_and_qrels(self, split):
        if self.hf_queries_repo:
            qrels_ds = load_dataset(
                self.hf_queries_repo,
                self.subset,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
                trust_remote_code=True,
            )[split]

        for nugget_list in qrels_ds["nuggets"]:
            for nugget in nugget_list:
                nugget_id = str(nugget["_id"])
                self.nuggets[str(nugget["_id"])] = nugget["text"]
                if nugget_id not in self.qrels_nuggets:
                    self.qrels_nuggets[nugget_id] = {}
                for doc_id in nugget["relevant_corpus_ids"]:
                    self.qrels_nuggets[nugget_id][doc_id] = 1
                for doc_id in nugget["non_relevant_corpus_ids"]:
                    self.qrels_nuggets[nugget_id][doc_id] = 0
