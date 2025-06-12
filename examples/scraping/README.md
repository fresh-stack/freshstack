# Examples on scraping questions from StackOverflow

### 1. Download StackOverflow Dumps

To download the StackOverflow data dump, i.e., `stackoverflow.com.7z`, you need to first have an account in StackOverflow. Click on your StackOverflow user profile, click on "Settings", then click on "Data dump access" and finally click on "Download data".
The page will view as follows:

```bash
Stack Overflow
Last uploaded: Apr 01, 2025
File size: 63.7 GB
```

### 2. Extract the 7z Stackoverflow files

Run the [unzip_stackoverflow.py](./unzip_stackoverflow.py) file to unzip and extract the XML Stack Overflow content.

### 3. Iterate through the StackOverflow XML file

Run the [parse_and_extract_queries.py][./parse_and_extract_queries.py] file to extract the stackoverflow queries and answers asked in Stackoverflow given a list of tags (or an individual tag alone).

### 4. Get top-k co-occuring tags for a given Stackoverflow tag

Run the [top_k_co_occuring_tags.py][/top_k_co_occuring_tags.py] file to get the top-k co-occuring tags for the given topic. This is especially useful if we want to find relevant github repositories for a given topic.