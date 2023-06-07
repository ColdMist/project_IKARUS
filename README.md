# Codename Project IKARUS
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: MIT](https://img.shields.io/badge/langchain-0.0.189-<green>)](https://python.langchain.com/en/latest/index.html)
[![License: MIT](https://img.shields.io/badge/OpenAI-0.27.7-<green>)](https://openai.com/)

This project AIMS to deliver the experience utilizing langchain library with OpenAI model to do several useful tasks
- [x] Question Answering from an external source of Knowledge base: 
  - [question_answering_from_knowledge_base.py](question_answering_from_knowledge_base.py) script leverages the power of langchain and OpenAI inorder to perform question answering from provided pdf as knowledge sources.
  - Restricts knowledge only to the sources provided in the folder where the pdfs are stored.
  - Utilizes ongoing conversation in memory buffer to perform question answering.
- [x] Natural Language to Structural Information
  - [natural_language_to_structured_info.py](natural_language_to_structured_info.py)  script converts natural language text into a structured graph, linking entities to their Wikidata instances, thereby creating a knowledge base for future use as a prompt.
  - the general workflow is as follows:
    - take a free text as input
    - build a graph from the free text
    - connect the entities to the real wikidata instances
    - collect the connected entities from wikidata per entity
    - build a structured knowledge source in order to later use as a prompt for further usage.
- [x] Question answering from structured file i.e, json.
    - [question_answering_with_reasoning_from_KB.py](question_answering_with_reasoning_from_KB.py) script performs chain reasoning to answer questions about knowledge base stored in JSON file structure. 

## Installation
## using pip
```commandline
pip install -r requirements.txt
```
## using conda
To begin with, one need to install the required software to operate the provided scripts. As a preliminary step, you must install the Conda environment, which you can find at https://docs.anaconda.com/anaconda/install/index.html.
After installation using conda command line, you can create a new environment using the following command:

```commandline
conda env create -f conda_env.yml
```

One need to activate the environment using the following command:

```commandline
conda activate ikarus
```

Download the required spacy model using the following command:

```commandline
python -m spacy download en_core_web_sm
```

## Usage
To run Question Answering from an external source of Knowledge base: 
```commandline
python question_answering_from_knowledge_base.py --api_key <YOUR OPENAI API KEY> --directory_path <The directory of all the stored PDFs>
```

To run Natural Language to Structural Information
```commandline
 python natural_language_to_structured_info.py --api_key <YOUR OPENAI API KEY> --text_or_filepath <The text (in natural language) or file_path>
```
Question answering from structured file i.e, json.
```commandline
python question_answering_with_reasoning_from_KB.py --api_key <YOUR OPENAI API KEY> --file_path <The file path of the JSON file>
```

