import base64

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import spacy
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load spaCy model for embeddings
nlp = spacy.load("en_core_web_sm")

# Ensure NLTK punkt is available for sentence tokenization
nltk.download('punkt')

# Paths where the model and tokenizer are saved
model_save_directory = "/home/.../GhostLink" # Update the directory location
tokenizer_save_directory = "/home/.../GhostLink" # Update the directory location

# Initialize the QA pipeline with the local model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_directory)
model = AutoModelForQuestionAnswering.from_pretrained(model_save_directory)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

documents_text = {}

app.layout = dbc.Container(fluid=True, children=[
    html.H1("GhostLink", className="text-center mb-4"),  # Title added here
    dbc.Row([
        dbc.Col(width=6, children=[
            html.H3("Document Summarizer"),
            dcc.Upload(id='upload-document', children=html.Div(['Drag and Drop or Click to Select Files']),
                       style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                              'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                       multiple=True),
            html.Div(id='file-checklist-container'),
            html.Button('Summarize Selected', id='summarize-btn', n_clicks=0, className="btn btn-primary mt-2"),
            html.Div(id='summary-output', className="mt-2")
        ]),
        dbc.Col(width=6, children=[
            html.H3("Query Response", className="text-center"),
            html.Div(id='chat-window', style={'height': '75vh', 'overflowY': 'auto'}),
            dcc.Input(id='user-query', type='text', placeholder='Enter your query here...', className="form-control"),
            html.Button('Submit', id='submit-query', n_clicks=0, className="btn btn-primary mt-2"),
        ]),
    ]),
])


@app.callback(
    Output('file-checklist-container', 'children'),
    Input('upload-document', 'contents'),
    State('upload-document', 'filename'),
    prevent_initial_call=True
)
def update_file_list(list_of_contents, list_of_names):
    if list_of_contents is None:
        raise PreventUpdate

    children = []
    for content, name in zip(list_of_contents, list_of_names):
        content_type, content_string = content.split(',')
        document_id = len(documents_text)
        document_key = f"doc-{document_id}"
        documents_text[document_key] = {'filename': name, 'content': base64.b64decode(content_string).decode('utf-8')}

        children.append(
            html.Div([
                dbc.Switch(id={'type': 'document-switch', 'index': document_key},
                           label=name,
                           className="form-check-input",
                           style={'border': 'none', 'backgroundColor': 'transparent'}),
            ], className="d-flex align-items-center mt-2", style={'backgroundColor': 'transparent', 'border': 'none'})
        )
    return children

@app.callback(
    Output('summary-output', 'children'),
    Input('summarize-btn', 'n_clicks'),
    State({'type': 'document-switch', 'index': dash.ALL}, 'value'),
    prevent_initial_call=True
)
def summarize_documents(n_clicks, switch_states):
    selected_docs_keys = [key for key, value in zip(documents_text.keys(), switch_states) if value]
    summaries = []
    for key in selected_docs_keys:
        doc = documents_text[key]
        summary_text = text_rank_summarization(doc['content'])
        summaries.append(html.P(f"{doc['filename']}: {summary_text}"))

    return summaries if summaries else "No documents selected or no content to summarize."

def text_rank_summarization(text, top_n_sentences=5):
    sentences = sent_tokenize(text)
    sentence_vectors = [nlp(sentence).vector for sentence in sentences]
    sim_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, -1), sentence_vectors[j].reshape(1, -1))[0, 0]

    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summarized_text = " ".join([s for _, s in ranked_sentences[:top_n_sentences]])
    return summarized_text

@app.callback(
    Output('chat-window', 'children'),
    Input('submit-query', 'n_clicks'),
    State('user-query', 'value'),
    State({'type': 'document-switch', 'index': dash.ALL}, 'value'),
    prevent_initial_call=True
)
def handle_query(n_clicks, user_query, switch_states):
    if not user_query:
        raise PreventUpdate

    selected_docs_keys = [key for key, value in zip(documents_text.keys(), switch_states) if value]
    context = " ".join(documents_text[key]['content'] for key in selected_docs_keys if key in documents_text)

    answer = qa_pipeline(question=user_query, context=context)

    return [html.P(f"Answer: {answer['answer']}")]

if __name__ == '__main__':
    app.run_server(debug=True)
