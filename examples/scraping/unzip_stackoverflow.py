"""
This script is used to extract the 7z files of stackoverflow.com and stackoverflow.com-Posts.7z.
To download stackoverflow.com.7z, you need to first have an account in StackOverflow.
Click on your StackOverflow user profile, click on "Settings", then click on "Data dump access" and finally click on "Download data".
The page will view as follows:

```
Stack Overflow
Last uploaded: Apr 01, 2025
File size: 63.7 GB
```
"""

import argparse
import logging
import os

from freshstack import LoggingHandler, util

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract StackOverflow 7z files")
    parser.add_argument(
        "--input_directory",
        type=str,
        required=True,
        help="Path to the 7z archive file to extract",
    )

    # # Extracting the outer level StackOverflow
    args = parser.parse_args()
    input_dir = args.input_directory
    archive_path = f"{input_dir}/stackoverflow.com.7z"
    extract_to = f"{input_dir}/stackoverflow.com"

    # make sure the directory exists
    os.makedirs(extract_to, exist_ok=True)
    logging.info(f"Extracting {archive_path} to {extract_to}...")
    util.extract_7z(archive_path, extract_to)

    # Extracting the inner level StackOverflow posts (stackoverflow.com-Posts.7z)
    output_dir = extract_to
    archive_path = os.path.join(output_dir, "stackoverflow.com-Posts.7z")
    extract_to = os.path.join(output_dir, "extracted")

    # make sure the directory exists
    os.makedirs(extract_to, exist_ok=True)
    logging.info(f"Extracting {archive_path} to {extract_to}...")
    util.extract_7z(archive_path, extract_to)
