"""
This script shows an example on how to chunk a GitHub repository end to end using our chunking code.
Make sure you have the requirements for chunking installed: `pip install freshstack[chunking]`.

Example script:
export KEYWORD="langchain"
export GITHUB_REPOSITORY=("langchain-ai/langchain")
export NAMES=("langchain")

export LOCAL_DIR="<your_local_dir>/raw"
export OUTPUT_DIR="<your_local_dir>/processed/"

# Get the number of repositories
num_repos=${#GITHUB_REPOSITORY[@]}

for ((i=0; i<num_repos; i++)); do
    repo=${GITHUB_REPOSITORY[$i]}
    keyword=$KEYWORD
    name=${NAMES[$i]}

    echo "Processing $repo with keyword $keyword"

    # Clone the repository and use the corresponding keyword
    NEW_OUTPUT_DIR="$OUTPUT_DIR/$keyword/"
    OUTPUT_FILENAME="corpus.$name.jsonl"
    python -m chunk_github_repo --repository_id $repo --local_dir $LOCAL_DIR --output_dir $NEW_OUTPUT_DIR --output_filename $OUTPUT_FILENAME
done
"""

import argparse
import logging

from freshstack import LoggingHandler
from freshstack.chunking import GitHubRepoChunker

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


def main():
    parser = argparse.ArgumentParser(description="Chunk a GitHub repository end to end")
    parser.add_argument("--repository_id", type=str, required=True, default="NovaSearch/stella_en_1.5B_v5")
    parser.add_argument("--local_dir", type=str, default=None, help="Local directory to store the repository")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./repository/",
        help="Batch size for encoding the queries and corpus documents",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="corpus.jsonl",
        help="The query prompt name to use for the model, default is 'default'",
    )
    parser.add_argument(
        "--included_extensions",
        type=set,
        default=None,
        nargs="*",
        help="Set of file extensions to include in chunking, e.g. ['.py', '.md', '.txt']",
    )
    parser.add_argument(
        "--excluded_extensions",
        type=set,
        default=None,
        nargs="*",
        help="Set of file extensions to exclude from chunking, e.g. ['.png', '.jpg', '.gif']",
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_chunks_allowed", type=int, default=100, required=False)
    parser.add_argument(
        "--max_chunk_characters",
        type=int,
        default=1000000,
        required=False,
        help="Maximum number of characters in a chunk, else Rust will panic",
    )
    args = parser.parse_args()

    # Set the local directory to store the repository
    logging.info(f"Starting to chunk the GitHub repository: {args.repository_id}")

    if args.excluded_extensions is None:
        logging.info("No excluded extensions provided, using default set of excluded extensions.")
        excluded_extensions = {".png", ".gif", ".bin", ".jpg", ".jpeg", ".mp4", ".csv", ".json"}
    else:
        excluded_extensions = set(args.excluded_extensions)

    # Initialize the GitHubRepoChunker object
    github_repo = GitHubRepoChunker(
        repo_id=args.repository_id,
        local_dir=args.local_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        included_extensions=args.included_extensions,
        excluded_extensions=excluded_extensions,
        max_tokens=args.max_tokens,
        max_chunks_allowed=args.max_chunks_allowed,
        max_chunk_characters=args.max_chunk_characters,
    )

    output_path = github_repo.chunk()
    logging.info(f"Chunked repository saved to: {output_path}")


if __name__ == "__main__":
    main()
