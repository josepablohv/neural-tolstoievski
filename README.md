# Tolstoy-Dostoevsky NLP Project

## Project Overview
This project aims to analyze and generate text in the distinct styles of Leo Tolstoy and Fyodor Dostoevsky, focusing on their Spanish translations. It will involve data cleaning, stylometric analysis, and text generation using modern NLP techniques.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/josepablohv/neural-tolstoievski.git
   cd tolstoy-dostoevsky-nlp
   ```

## Requirements
Install the necessary libraries:
```bash
pip install -r requirements.txt
```

## Notebooks
- `notebooks/01_data_cleaning.ipynb`: data loading, profiling and cleaning steps used to produce the cleaned texts.
- `notebooks/04_language_model_feeding.ipynb`: preparing tokenized data and feeding examples to language models.
- `notebooks/05_author_encoder_classifier.ipynb`: experiments with author encoding and classification.
- `notebooks/06_rag_qa_model.ipynb`: LangChain RAG Q&A demo — this notebook and the small Streamlit app in `streamlit_app/` belong to the same RAG/QA mini-project.

Keep other notebooks for exploration; the four listed above are the main entry points for the current pipeline and RAG demo.