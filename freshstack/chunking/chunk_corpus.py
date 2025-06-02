"""Module for chunking GitHub repositories into smaller text segments."""
import json
import os
import pathlib
import re

from chunker import TokenizerFunc, UniversalFileChunker
from data_manager import GitHubRepoManager
from tqdm.autonotebook import tqdm


class GitHubRepoChunker:
    """
    A class for downloading and chunking GitHub repositories into manageable text segments.
    """
    
    def __init__(
        self,
        repo_id: str,
        local_dir: str,
        output_dir: str,
        tokenizer: TokenizerFunc | None = None,
        output_filename: str = "corpus.jsonl",
        included_extensions: set[str] | None = None,
        excluded_extensions: set[str] | None = None,
        max_tokens: int = 2048,
        max_chunks_allowed: int = 100,
        max_chunk_characters: int = 1000000
    ):
        """
        Initialize the GitHubRepoChunker.

        Args:
            repo_id: The GitHub repository ID (e.g., "langchain-ai/langchain")
            local_dir: The local directory to download the repository to
            output_dir: The directory to output chunked files to
            output_filename: The name of the output file
            included_extensions: File extensions to include (None means include all)
            excluded_extensions: File extensions to exclude
            max_tokens: Maximum number of tokens per chunk
            max_chunks_allowed: Maximum number of chunks allowed per file
            max_chunk_characters: Maximum number of characters in a chunk
        """
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.output_dir = output_dir
        self.output_filename = output_filename

        if excluded_extensions is None:
            self.excluded_extensions = {".png", ".gif", ".bin", ".jpg", ".jpeg", ".mp4", ".csv", ".json"}
        else:
            self.excluded_extensions = excluded_extensions

        self.included_extensions = included_extensions
        self.max_tokens = max_tokens
        self.max_chunks_allowed = max_chunks_allowed
        self.max_chunk_characters = max_chunk_characters
        
        # Initialize components
        self.github_repo = GitHubRepoManager(
            repo_id=self.repo_id,
            local_dir=self.local_dir,
            included_extensions=self.included_extensions,
            excluded_extensions=self.excluded_extensions
        )
        
        self.chunker = UniversalFileChunker(max_tokens=self.max_tokens)
    
    def _clean_content(self, content: str) -> str:
        """
        Clean the content by removing binary data embedded in text.
        
        Args:
            content: The content to clean
            
        Returns:
            Cleaned content
        """
        # Remove image data (png/jpeg) from jupyter notebook files
        if any(keyword in content for keyword in ["image/png", "image/jpeg"]):
            for _ in range(3):
                pattern = r'"(?:image\/png|image\/jpeg)":\s*"([^"]*)"'
                content = re.sub(pattern, '""', content, re.DOTALL)
        
        # Remove base64 encoded data (video/mp4, image/jpeg, image/png, audio/x-wav)
        if any(keyword in content for keyword in [
            "data:video/mp4;base64,", 
            "data:image/jpeg;base64,",
            "data:image/png;base64,", 
            "data:audio/x-wav;base64"
        ]):
            for _ in range(3):
                pattern = r'"data:(?:video/mp4|image/jpeg|image/png|audio/x-wav);base64,([^"]+)"'
                content = re.sub(pattern, '""', content, re.DOTALL)
                
        return content
    
    def download(self) -> None:
        """Download the GitHub repository"""
        self.github_repo.download()
    
    def process(self) -> str:
        """
        Process the repository, chunking all files and writing to output.
        
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_filename)
        
        # Get approximate file count for progress bar
        github_path = pathlib.Path(os.path.join(self.local_dir, self.repo_id))
        num_files = sum(1 for _ in github_path.rglob("*") if _.is_file())
        
        # Process all files
        with open(output_path, 'w') as fout:
            for content, metadata in tqdm(self.github_repo.walk(), total=num_files, desc="Chunking files"):
                # Clean content
                content = self._clean_content(content)
                
                # Skip if content is too large
                if len(content) >= self.max_chunk_characters:
                    continue
                
                # Chunk the content
                try:
                    chunks = self.chunker.chunk(content, metadata)
                    
                    # Only process if within chunk limit
                    if len(chunks) <= self.max_chunks_allowed:
                        for chunk in chunks:
                            chunk_text = chunk.file_content[chunk.start_byte:chunk.end_byte]
                            
                            # Skip empty chunks
                            if chunk_text.strip() == "":
                                continue
                                
                            # Create document
                            document = {
                                "_id": chunk.metadata['id'].replace(self.repo_id + "/", ""),
                                "title": "",
                                "text": chunk_text,
                                "metadata": {
                                    "url": chunk.file_metadata['url'],
                                    "start_byte": chunk.start_byte,
                                    "end_byte": chunk.end_byte,
                                }
                            }
                            
                            # Write to output
                            fout.write(json.dumps(document, ensure_ascii=False) + '\n')
                            fout.flush()
                
                except Exception as e:
                    print(f"Error chunking content: {e}")
                    continue
        
        return output_path
    
    def run(self) -> str:
        """
        Download and process the repository.
        
        Returns:
            Path to the output file
        """
        self.download()
        return self.process()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Chunk GitHub repositories into text segments")
    parser.add_argument("--repo_id", type=str, default="langchain-ai/langchain", help="GitHub repository ID")
    parser.add_argument("--local_dir", type=str, default="/tmp/", help="Local directory to download to")
    parser.add_argument("--included_extensions", type=str, nargs="*", help="File extensions to include")
    parser.add_argument("--excluded_extensions", type=str, nargs="*", help="File extensions to exclude")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens per chunk")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer function (optional)")
    parser.add_argument("--output_dir", type=str, default="/tmp/", help="Output directory")
    parser.add_argument("--output_filename", type=str, default="corpus.jsonl", help="Output filename")
    parser.add_argument("--max_chunks_allowed", type=int, default=100, help="Maximum chunks allowed per file")
    parser.add_argument("--max_chunk_characters", type=int, default=1000000, 
                        help="Maximum characters in a chunk, else Rust will panic")

    args = parser.parse_args()

    # Convert list arguments to sets if provided
    included_extensions = set(args.included_extensions) if args.included_extensions else None
    excluded_extensions = set(args.excluded_extensions) if args.excluded_extensions else \
        {".png", ".gif", ".bin", ".jpg", ".jpeg", ".mp4", ".csv", ".json"}

    # Initialize and run the chunker
    chunker = GitHubRepoChunker(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        tokenizer=args.tokenizer,
        included_extensions=included_extensions,
        excluded_extensions=excluded_extensions,
        max_tokens=args.max_tokens,
        max_chunks_allowed=args.max_chunks_allowed,
        max_chunk_characters=args.max_chunk_characters
    )

    output_path = chunker.run()
    print(f"Repository chunking complete. Output written to: {output_path}")
