"""
An data example of the April 2024 dump of StackExchange
{'Id': '38779', 'PostTypeId': '1', 'AcceptedAnswerId': '40472', 'CreationDate': '2008-09-02T03:41:06.880', 'Score': '6', 'ViewCount': '6282'
'Body': "<p>I have a wcf application hosted in a windows service running a local windows account. Do I need to set an SPN for this account? If so, what's the protocol the SPN needs to be set under? I know how to do this for services over HTTP, but have never done it for net.tcp.</p>\n",
'OwnerUserId': '781', 'OwnerDisplayName': 'Esteban', 'LastEditorUserId': '1116', 'LastEditorDisplayName': 'John Nolan', 'LastEditDate': '2008-09-02T08:49:19.323', 'LastActivityDate': '2013-06-24T17:03:55.833',
'Title': 'What SPN do I need to set for a net.tcp service?',
'Tags': '|wcf|security|spn|', 'AnswerCount': '2', 'CommentCount': '0', 'FavoriteCount': '0', 'ContentLicense': 'CC BY-SA 2.5'}

Usage:
export OUTPUT_DIR=<your_output_dir>
python parse_and_extract_queries.py \
    --input_xml "${OUTPUT_DIR}/Posts.xml" \
    --keywords "laravel-10,laravel-11" \
    --filter "tags" \
    --output_dir "${OUTPUT_DIR}/extracted" \
    --output_filename "Posts.laravel.jsonl"
"""

import argparse
import json
import logging
import os
import re

from bs4 import BeautifulSoup
from lxml import etree
from tqdm.autonotebook import tqdm

from freshstack import LoggingHandler

# Just some code to print debug information to stdout
logger = logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def skip_exceptions(it):
    while True:
        try:
            yield next(it)
        except StopIteration:
            break
        except Exception as e:
            logger.warning(f"Skipping iteration because of exception {e}")


def parse_xml(
    xml_file: str,
    jsonl_file: str,
    keywords: list[str],
    filter: str = "all" | "title" | "body" | "tags",
    total: int = 119999898,
):
    f = open(xml_file, "rb")
    xmlit = iter(etree.iterparse(f, events=("start", "end"), recover=True))
    queries = {}

    for event, elem in tqdm(skip_exceptions(xmlit), total=total, desc="parsing XML to get queries...."):
        if event == "end":
            # Process each element as required
            if elem.tag == "row":
                # check if the post is a question
                if int(elem.attrib["PostTypeId"]) == 1:  # question
                    if filter == "all":
                        search_text = (
                            elem.attrib["Body"].lower()
                            + " "
                            + elem.attrib["Title"].lower()
                            + " "
                            + elem.attrib.get("Tags", "").lower()
                        )
                    elif filter == "title":
                        search_text = elem.attrib["Title"].lower()
                    elif filter == "body":
                        search_text = elem.attrib["Body"].lower()
                    elif filter == "tags":
                        search_text = elem.attrib.get("Tags", "").lower()

                    # check if the keyword is present in the search text
                    if any(keyword in search_text for keyword in keywords):
                        row = {}
                        soup = BeautifulSoup(elem.attrib["Body"], "html.parser")
                        body_text = soup.get_text()
                        tags = re.findall(r"<(.*?)>", elem.attrib.get("Tags", ""))
                        row["id"] = elem.attrib["Id"]
                        row["title"] = elem.attrib["Title"]
                        row["body"] = body_text
                        row["tags"] = tags
                        row["score"] = elem.attrib["Score"]
                        row["views"] = elem.attrib["ViewCount"]
                        row["date"] = elem.attrib["CreationDate"]
                        row["accepted_answer"] = elem.attrib.get("AcceptedAnswerId", "")
                        row["answers"] = {}
                        row["metadata"] = dict(elem.attrib)
                        queries[elem.attrib["Id"]] = row

                elem.clear()  # Clear the element from memory to save space

                # logging 100 queries
                if 100 <= len(queries) < 1000 and len(queries) % 100 == 0:
                    logger.info(f"Accumulated {len(queries)} queries...")

                if 1000 <= len(queries) < 10000 and len(queries) % 1000 == 0:
                    logger.info(f"Accumulated {len(queries)} queries...")

                if 10000 <= len(queries) and len(queries) % 10000 == 0:
                    logger.info(f"Accumulated {len(queries)} queries...")

    query_ids = set(queries.keys())

    # Running one more time to get the accepted answers
    f = open(xml_file, "rb")
    xmlit = iter(etree.iterparse(f, events=("start", "end"), recover=True))

    for event, elem in tqdm(skip_exceptions(xmlit), total=total, desc="parsing XML file to get all answers...."):
        if event == "end":
            # Process each element as required
            if elem.tag == "row":
                # check if the post is an answer to a question
                if int(elem.attrib["PostTypeId"]) == 2:  # answer
                    if elem.attrib["ParentId"] in query_ids:
                        row = {}
                        soup = BeautifulSoup(elem.attrib["Body"], "html.parser")
                        body_text = soup.get_text()
                        row["id"] = elem.attrib["Id"]
                        row["body"] = body_text
                        row["score"] = elem.attrib["Score"]
                        row["date"] = elem.attrib["CreationDate"]
                        row["user_id"] = elem.attrib.get("OwnerUserId", "")
                        row["username"] = elem.attrib.get("OwnerDisplayName", "")
                        row["metadata"] = dict(elem.attrib)
                        queries[elem.attrib["ParentId"]]["answers"][elem.attrib["Id"]] = row

                elem.clear()  # Clear the element from memory to save space

    # Save the queries as a jsonl file
    with open(jsonl_file, "w") as f:
        for _, item in queries.items():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract query-answer pairs from StackExchange dump")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--input_xml", type=str, help="Path to the 7z archive")
    parser.add_argument("--keywords", type=str, help="Keywords to filter")
    parser.add_argument("--filter", type=str, default="tags", help="Filter to apply on the search text")
    parser.add_argument("--output_filename", type=str, default="Posts.huggingface.v2.jsonl", help="Output filename")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # All the top 2023-2024 keywords
    # keywords = ['langchain', 'laravel-10', 'godot4', 'chatgpt-api', 'swiftdata', 'next.js14', 'angular17', 'yolov8', 'azure-openai', 'py-langchain', 'llama',
    #             'xcode15', 'llama-index', 'ios17', 'tanstackreact-query', 'visionos', 'angular16', 'apache-age', 'flutterflow', 'chromadb', 'expo-router', 'shadcnui',
    #             'filamentphp', 'laravel-11', 'java-21', 'macos-sonoma', 'supabase-js', 'timefold', 'asp.net-core-8', 'azure-synapse-analytics', 'tanstack', 'google-gemini',
    #             'android-14', 'nextjs14', 'neovim-plugin', 'trpc', 'nx-monorepo', 'python-3.12', 'microsoft-fabric', 'phaserjs', 'railway', 'gpt-4', 'angular-signals', 'msal',
    #             'tms-web-core', 'ollama', 'pinecone', 'retrieval-augmented-generation', 'odoo-17', 'otel', 'unreal', 'fullcalendar-6', 'ionic7', 'ef-core-8.0', 'openaiembeddings',
    #             'next-intl', 'server-action', 'pydantic-v2', 'huggingface-trainer', 'delphi-12-athens', 'yolov7', 'material-react-table', 'radix-ui', 'livewire-3', 'vaadin24',
    #             'agora', 'read-csv', 'maui-android', 'typo3-12.x', 'google-generativeai', 'supabase-flutter', 'amazon-bedrock', 'mistral-7b', 'wagmi', 'peft', 'vector-search',
    #             'drupal-10', 'langchain-js', 'chat-gpt-4', 'netsuite-rest-api', 'azure-synapse-pipeline', 'postgresql-16', 'typst', 'tanstack-table', 'pgvector', 'huggingface-hub',
    #             'nativewind', 'widgetliveactivity', 'llama-cpp-python', 'auto-route', 'ultralytics', 'exchange-online', 'azure-ai-search', 'llamacpp', 'catalystbyzoho', 'angular18']

    keywords = args.keywords.split(",")
    output_filename = os.path.join(args.output_dir, args.output_filename)
    parse_xml(args.input_xml, output_filename, keywords=keywords, filter=args.filter)


if __name__ == "__main__":
    main()
