"""
Chunker abstraction and implementations.
Implementation originally taken from https://github.com/Storia-AI/repo2vec.
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from traitlets.config import Config
from transformers import AutoTokenizer

from .util import parse_notebook_blob, remove_embedded_media

if importlib.util.find_spec("nbconvert") is not None:
    from nbconvert import ScriptExporter

# Ensure required libraries are installed
if importlib.util.find_spec("pygments") is not None:
    import pygments  # type: ignore[import]

if importlib.util.find_spec("tiktoken") is not None:
    import tiktoken  # type: ignore[import]

    DEFAULT_TOKENIZER = tiktoken.get_encoding("cl100k_base")
else:
    DEFAULT_TOKENIZER = None

if importlib.util.find_spec("semchunk") is not None:
    from semchunk import chunk as chunk_via_semchunk  # type: ignore

if importlib.util.find_spec("tree_sitter") is not None:
    from tree_sitter import Node  # type: ignore[import]

if importlib.util.find_spec("tree_sitter_language_pack") is not None:
    from tree_sitter_language_pack import get_parser  # type: ignore[import]

logger = logging.getLogger(__name__)

# Type for tokenizer functions
TokenizerFunc = Callable[[str], list]


class Chunk:
    """Abstract class for a chunk of code or text to be indexed."""

    @abstractmethod
    def content(self) -> str:
        """The content of the chunk to be indexed."""

    @abstractmethod
    def metadata(self) -> dict:
        """Metadata for the chunk to be indexed."""


@dataclass
class FileChunk(Chunk):
    """A chunk of code or text extracted from a file in the repository."""

    file_content: str  # The content of the entire file, not just this chunk.
    file_metadata: dict  # Metadata of the entire file, not just this chunk.
    start_byte: int
    end_byte: int
    tokenizer_func: TokenizerFunc = None  # Function to count tokens

    def __post_init__(self):
        if self.tokenizer_func is None:
            self.tokenizer_func = lambda x: DEFAULT_TOKENIZER.encode(x, disallowed_special=())

    @cached_property
    def filename(self):
        if "file_path" not in self.file_metadata:
            raise ValueError("file_metadata must contain a 'file_path' key.")
        return self.file_metadata["file_path"]

    @cached_property
    def content(self) -> str | None:
        """The text content to be embedded. Might contain information beyond just the text snippet from the file."""
        return self.filename + "\n\n" + self.file_content[self.start_byte : self.end_byte]

    @cached_property
    def metadata(self):
        """Converts the chunk to a dictionary that can be passed to a vector store."""
        # Some vector stores require the IDs to be ASCII.
        filename_ascii = self.filename.encode("ascii", "ignore").decode("ascii")
        chunk_metadata = {
            # Some vector stores require the IDs to be ASCII.
            "id": f"{filename_ascii}_{self.start_byte}_{self.end_byte}",
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            # Note to developer: When choosing a large chunk size, you might exceed the vector store's metadata
            # size limit. In that case, you can simply store the start/end bytes above, and fetch the content
            # directly from the repository when needed.
            "text": self.content,
        }
        chunk_metadata.update(self.file_metadata)
        return chunk_metadata

    @cached_property
    def num_tokens(self):
        """Number of tokens in this chunk."""
        tokens = self.tokenizer_func(self.content)
        # 1) If someone already returned an integer count
        if isinstance(tokens, int):
            return tokens
        # 2) If tokenizer returns a flat list of token IDs
        if isinstance(tokens, list):
            return len(tokens)
        # 3) If tokenizer returns a HF-style dict with 'input_ids'
        if isinstance(tokens, dict) and "input_ids" in tokens:
            return len(tokens["input_ids"])
        # Otherwise, we can’t figure it out
        raise ValueError(
            "Tokenizer function must return an int (count), a list of tokens, or a dict with 'input_ids'."
        )

    def __eq__(self, other):
        if isinstance(other, Chunk):
            return (
                self.filename == other.filename
                and self.start_byte == other.start_byte
                and self.end_byte == other.end_byte
            )
        return False

    def __hash__(self):
        return hash((self.filename, self.start_byte, self.end_byte))


class Chunker(ABC):
    """Abstract class for chunking a datum into smaller pieces."""

    @abstractmethod
    def chunk(self, content, metadata: dict) -> list[Chunk]:
        """Chunks a datum into smaller pieces."""


class CodeFileChunker(Chunker):
    """Splits a code file into chunks of at most `max_tokens` tokens each."""

    def __init__(self, max_tokens: int, tokenizer_func: TokenizerFunc | None = None):
        self.max_tokens = max_tokens
        self.tokenizer_func = tokenizer_func
        self.text_chunker = TextFileChunker(max_tokens, tokenizer_func)

    @staticmethod
    def _get_language_from_filename(filename: str):
        """Returns a canonical name for the language of the file, based on its extension.
        Returns None if the language is unknown to the pygments lexer.
        """
        try:
            lexer = pygments.lexers.get_lexer_for_filename(filename)
            return lexer.name.lower()
        except pygments.util.ClassNotFound:
            return None

    def _chunk_node(self, node: Node, file_content: str, file_metadata: dict) -> list[FileChunk]:
        """Splits a node in the parse tree into a flat list of chunks."""
        node_chunk = FileChunk(file_content, file_metadata, node.start_byte, node.end_byte, self.tokenizer_func)

        if node_chunk.num_tokens <= self.max_tokens:
            return [node_chunk]

        if not node.children:
            # This is a leaf node, but it's too long. We'll have to split it with a text tokenizer.
            return self.text_chunker.chunk(file_content[node.start_byte : node.end_byte], file_metadata)

        chunks = []
        for child in node.children:
            chunks.extend(self._chunk_node(child, file_content, file_metadata))

        for chunk in chunks:
            # This should always be true. Otherwise there must be a bug in the code.
            assert chunk.num_tokens <= self.max_tokens

        # Merge neighboring chunks if their combined size doesn't exceed max_tokens. The goal is to avoid pathologically
        # small chunks that end up being undeservedly preferred by the retriever.
        merged_chunks = []
        for chunk in chunks:
            if not merged_chunks:
                merged_chunks.append(chunk)
            elif merged_chunks[-1].num_tokens + chunk.num_tokens < self.max_tokens - 50:
                # There's a good chance that merging these two chunks will be under the token limit. We're not 100% sure
                # at this point, because tokenization is not necessarily additive.
                merged = FileChunk(
                    file_content, file_metadata, merged_chunks[-1].start_byte, chunk.end_byte, self.tokenizer_func
                )
                if merged.num_tokens <= self.max_tokens:
                    merged_chunks[-1] = merged
                else:
                    merged_chunks.append(chunk)
            else:
                merged_chunks.append(chunk)
        chunks = merged_chunks

        for chunk in merged_chunks:
            # This should always be true. Otherwise there's a bug worth investigating.
            assert chunk.num_tokens <= self.max_tokens

        return merged_chunks

    @staticmethod
    def is_code_file(filename: str) -> bool:
        """Checks whether pygment & tree_sitter can parse the file as code."""
        language = CodeFileChunker._get_language_from_filename(filename)
        return language and language not in ["text only", "None"]

    @staticmethod
    def parse_tree(filename: str, content: str) -> list[str]:
        """Parses the code in a file and returns the parse tree."""
        language = CodeFileChunker._get_language_from_filename(filename)

        if not language or language in ["text only", "None"]:
            logging.debug("%s doesn't seem to be a code file.", filename)
            return None

        try:
            parser = get_parser(language)

        except LookupError:
            logging.debug("%s doesn't seem to be a code file.", filename)
            return None

        tree = parser.parse(bytes(content, "utf8"))

        if not tree.root_node.children or tree.root_node.children[0].type == "ERROR":
            logging.warning("Failed to parse code in %s.", filename)
            return None
        return tree

    def chunk(self, content, metadata: dict) -> list[Chunk]:
        """Chunks a code file into smaller pieces."""
        file_content = content
        file_metadata = metadata
        file_path = metadata["file_path"]

        if not file_content.strip():
            return []

        tree = self.parse_tree(file_path, file_content)
        if tree is None:
            return []

        file_chunks = self._chunk_node(tree.root_node, file_content, file_metadata)
        for chunk in file_chunks:
            # Make sure that the chunk has content and doesn't exceed the max_tokens limit. Otherwise there must be
            # a bug in the code.
            assert (
                chunk.num_tokens <= self.max_tokens
            ), f"Chunk size {chunk.num_tokens} exceeds max_tokens {self.max_tokens}."

        return file_chunks


class TextFileChunker(Chunker):
    """Wrapper around semchunk: https://github.com/umarbutler/semchunk."""

    def __init__(self, max_tokens: int, tokenizer_func: TokenizerFunc | None = None):
        self.max_tokens = max_tokens

        # Set the tokenizer function
        if tokenizer_func:
            # tokenizer_func *is* the function that returns a list of tokens (or HF dict)
            def count_tokens(text: str) -> int:
                tokens = tokenizer_func(text)
                if isinstance(tokens, list):
                    return len(tokens)
                elif isinstance(tokens, dict) and "input_ids" in tokens:
                    return len(tokens["input_ids"])
                else:
                    raise ValueError("TokenizerFunc must return a list or a dict with 'input_ids'")

            self.count_tokens = count_tokens
        else:
            self.count_tokens = lambda text: len(DEFAULT_TOKENIZER.encode(text, disallowed_special=()))

    def chunk(self, content, metadata: dict) -> list[Chunk]:
        """Chunks a text file into smaller pieces."""
        file_content = content
        file_metadata = metadata
        file_path = file_metadata["file_path"]

        # We need to allocate some tokens for the filename, which is part of the chunk content.
        extra_tokens = self.count_tokens(file_path + "\n\n")
        text_chunks = chunk_via_semchunk(file_content, self.max_tokens - extra_tokens, self.count_tokens)

        file_chunks = []
        start = 0
        for text_chunk in text_chunks:
            # This assertion should always be true. Otherwise there's a bug worth finding.
            assert self.count_tokens(text_chunk) <= self.max_tokens - extra_tokens

            # Find the start/end positions of the chunks.
            start = file_content.find(text_chunk, start)
            if start == -1:
                logging.warning("Couldn't find semchunk in content: %s", text_chunk)
            else:
                end = start + len(text_chunk)

                # Pass the tokenizer function to the FileChunk
                def tokenizer_func(x):
                    return self.count_tokens(x)

                file_chunks.append(FileChunk(file_content, file_metadata, start, end, tokenizer_func))

            start = end if start != -1 else 0

        return file_chunks


class IpynbFileChunker(Chunker):
    """Extracts the python code from a Jupyter notebook, removing all the boilerplate.

    Based on https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_retrieval_augmented_generation.ipynb
    """

    def __init__(self, code_chunker: CodeFileChunker):
        self.code_chunker = code_chunker
        self.c = Config()
        self.c.TemplateExporter.exclude_raw = True  # this skips all cell_type=="raw"
        self.exporter = ScriptExporter(config=self.c)

    def chunk(self, content, metadata: dict) -> list[Chunk]:
        filename = metadata["file_path"]
        if not filename.lower().endswith(".ipynb"):
            logging.warning("IPYNBChunker is only for .ipynb files.")
            return []

        python_code, _ = self.exporter.from_notebook_node(parse_notebook_blob(content))
        python_code = remove_embedded_media(python_code)  # Remove embedded media like images
        python_code = (
            python_code.replace("%pip", "#%pip").replace("!pip", "#!pip").replace("pip", "#pip")
        )  # Convert magic commands to comments (Important for chunking)
        metadata["file_path"] = filename.replace(".ipynb", ".py")  # Change the file extension to .py for code chunking

        # Now hand off to the code chunker
        chunks = self.code_chunker.chunk(
            content=python_code,
            metadata=metadata,  # Use .py for code chunking
        )
        # restore the .ipynb name
        for c in chunks:
            c.metadata["id"] = c.metadata["id"].replace(".py", ".ipynb")
            c.file_metadata["file_path"] = c.file_metadata["file_path"].replace(".py", ".ipynb")
        return chunks


class UniversalFileChunker(Chunker):
    """Chunks a file into smaller pieces, regardless of whether it's code or text."""

    def __init__(self, max_tokens: int, tokenizer: TokenizerFunc | None = None):
        """
        Initializes the UniversalFileChunker with a maximum token limit.

        Args:
            max_tokens: Maximum number of tokens per chunk
            tokenizer: Optional custom tokenizer. Can be either:
                       - A string name for tiktoken encodings (e.g. "cl100k_base")
                       - A function that takes text and returns a list of tokens
                       - A tokenizer object with an encode method
        """
        if tokenizer is None:
            logging.info(
                "Initializing UniversalFileChunker with max_tokens=%d with default tokenizer: cl100k_base", max_tokens
            )

            # Default tokenizer
            def tokenizer_func(x):
                return DEFAULT_TOKENIZER.encode(x, disallowed_special=())
        elif isinstance(tokenizer, str):
            logging.info(f"Using tiktoken encoding: {tokenizer}")
            # String identifier for tiktoken
            tk = tiktoken.get_encoding(tokenizer)

            def tokenizer_func(x):
                return tk.encode(x, disallowed_special=())
        elif hasattr(tokenizer, "encode"):
            logging.info("Using custom tokenizer with encode method with HF.")

            # Has encode method (like HF tokenizers)
            def tokenizer_func(x):
                return tokenizer.encode(x)
        else:
            logging.warning(f"Unrecognized tokenizer type: {type(tokenizer)}. Using default.")

            def tokenizer_func(x):
                return DEFAULT_TOKENIZER.encode(x, disallowed_special=())

        # Initialize chunkers with our tokenizer function
        self.code_chunker = CodeFileChunker(max_tokens, tokenizer_func)
        self.text_chunker = TextFileChunker(max_tokens, tokenizer_func)
        self.ipynb_chunker = IpynbFileChunker(self.code_chunker)

    def chunk(self, content, metadata: dict) -> list[Chunk]:
        if "file_path" not in metadata:
            raise ValueError("metadata must contain a 'file_path' key.")
        file_path = metadata["file_path"]

        # Figure out the appropriate chunker to use.
        if file_path.lower().endswith(".ipynb"):
            chunker = self.ipynb_chunker
        elif CodeFileChunker.is_code_file(file_path):
            chunker = self.code_chunker
        else:
            chunker = self.text_chunker

        return chunker.chunk(content, metadata)


if __name__ == "__main__":
    metadata = {"file_path": "example.py"}
    content = """import py7zr
    from lxml import etree
    import os, json, re
    from bs4 import BeautifulSoup
    from tqdm.autonotebook import tqdm
    from typing import Literal, list
    import argparse
    """

    # Example usage with default tokenizer
    chunker1 = UniversalFileChunker(max_tokens=50)  # default tokenizer is tiktoken's cl100k_base
    chunks = chunker1.chunk(content, metadata)
    print("Chunks using default tokenizer: cl100k_base with max_tokens=50")
    for chunk in chunks:
        print(chunk.metadata["id"], chunk.file_content[chunk.start_byte : chunk.end_byte])

    # Example usage with custom tiktoken encoding
    chunker2 = UniversalFileChunker(max_tokens=50, tokenizer="o200k_base")
    chunks = chunker2.chunk(content, metadata)
    print("\nChunks using custom tokenizer: o200k_base with max_tokens=50")
    for chunk in chunks:
        print(chunk.metadata["id"], chunk.file_content[chunk.start_byte : chunk.end_byte])

    # Example with HuggingFace tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    chunker3 = UniversalFileChunker(max_tokens=50, tokenizer=hf_tokenizer)
    chunks = chunker3.chunk(content, metadata)
    print("\nChunks using HuggingFace gpt2 tokenizer with max_tokens=50")
    for chunk in chunks:
        print(chunk.metadata["id"], chunk.file_content[chunk.start_byte : chunk.end_byte])
