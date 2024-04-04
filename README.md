# GhostLink

## Overview

GhostLink is a Python application designed to provide document summarization and query response functionalities using natural language processing (NLP) techniques. It leverages several libraries such as Dash, spaCy, NLTK, Transformers, and NetworkX to provide a seamless user experience in summarizing documents and responding to user queries.

## Features

### Document Summarizer

The Document Summarizer feature allows users to upload multiple documents and generate summaries based on the content of these documents. It utilizes TextRank algorithm for automatic summarization.

### Query Response

The Query Response feature enables users to input queries related to the uploaded documents. GhostLink then retrieves relevant information from the documents to answer the queries using a pre-trained question-answering model.

## Script Breakdown

### Import Statements

```python
import base64
import dash
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import spacy
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import PyTorch
```

- The script begins with importing necessary libraries and modules.
- `base64`: Used for decoding base64-encoded content.
- `dash`: A Python framework for building web applications.
- `spacy`: Industrial-strength natural language processing library.
- `networkx`: Library for studying complex networks.
- `numpy`: Library for numerical computing.
- `sklearn.metrics.pairwise`: Pairwise metrics for calculating similarity.
- `nltk`: Natural Language Toolkit for NLP tasks.
- `transformers`: Library for state-of-the-art natural language processing.

### Initialize NLP Models and Paths

```python
nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')

model_save_directory = "/home/.../GhostLink"
tokenizer_save_directory = "/home/.../GhostLink"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_directory)
model = AutoModelForQuestionAnswering.from_pretrained(model_save_directory)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
```

- Initialize spaCy model for word embeddings.
- Download NLTK punkt tokenizer.
- Define paths for saving the question-answering model and tokenizer.
- Load the question-answering model and tokenizer using Hugging Face's Transformers library.

### Initialize Dash Application and Layout

```python
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

documents_text = {}

app.layout = dbc.Container(fluid=True, children=[
    ...
])
```

- Create a Dash application instance.
- Define the layout using Dash Bootstrap Components for styling.

### Callback Functions

#### `update_file_list`

```python
@app.callback(
    Output('file-checklist-container', 'children'),
    Input('upload-document', 'contents'),
    State('upload-document', 'filename'),
    prevent_initial_call=True
)
def update_file_list(list_of_contents, list_of_names):
    ...
```

- Updates the list of uploaded documents when new documents are uploaded.

#### `summarize_documents`

```python
@app.callback(
    Output('summary-output', 'children'),
    Input('summarize-btn', 'n_clicks'),
    State({'type': 'document-switch', 'index': dash.ALL}, 'value'),
    prevent_initial_call=True
)
def summarize_documents(n_clicks, switch_states):
    ...
```

- Summarizes selected documents using the TextRank algorithm.

#### `handle_query`

```python
@app.callback(
    Output('chat-window', 'children'),
    Input('submit-query', 'n_clicks'),
    State('user-query', 'value'),
    State({'type': 'document-switch', 'index': dash.ALL}, 'value'),
    prevent_initial_call=True
)
def handle_query(n_clicks, user_query, switch_states):
    ...
```

- Handles user queries and retrieves answers from the uploaded documents using the question-answering model.

## Usage

To run GhostLink, execute the provided Python script (`ghostlink.py`). Ensure that all dependencies are installed before running the script.

```bash
python ghostlink.py
```

Access GhostLink through a web browser at `http://localhost:8050` after the server starts.

### Document Summarizer

1. **Upload Documents**: Click on the "Upload Documents" button and select one or more documents from your local file system.

2. **Summarize Selected**: After uploading documents, click the "Summarize Selected" button to generate summaries for the uploaded documents.

### Query Response

1. **Enter Query**: Type your query in the input field provided under the "Query Response" section.

2. **Submit**: Click on the "Submit" button to submit your query.

## Contributing

Contributions to GhostLink are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/your/repository).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dash: A Python framework for building analytical web applications.
- spaCy: Industrial-strength natural language processing library in Python.
- NLTK: A leading platform for building Python programs to work with human language data.
- Transformers: State-of-the-art natural language processing for PyTorch and TensorFlow.
- NetworkX: A Python library for studying complex networks.
