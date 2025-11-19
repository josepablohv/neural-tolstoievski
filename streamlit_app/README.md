# Streamlit App — Tolstoi Project

Minimal Streamlit app and utilities for vectorizing book text and serving a simple UI.

## Contents
- `hello_world_app.py` — simple Streamlit example app.
- `vectorize_files.py` — split text into chunks, compute embeddings, and persist a Chroma vector store.
- `data/` (expected) — cleaned book text files.
- `vectors/` (output) — persistent vector stores created by `vectorize_files.py`.

## Requirements
- Python 3.9+
- Recommended: virtual environment (venv / conda)
- Key packages (example): streamlit, python-dotenv, langchain, langchain_community, langchain_chroma, langchain_google_genai, tqdm

## Environment
Create a `.env` with required API keys (e.g. `GOOGLE_API_KEY=...`). The scripts load env vars via python-dotenv.

## Run the Streamlit app
From this folder:
```powershell
streamlit run hello_world_app.py
```

## Vectorizing books
Build vectors for a single book:
```powershell
python vectorize_files.py --book guerra_y_paz
```
Force re-vectorize:
```powershell
python vectorize_files.py --book guerra_y_paz --force
```

## Quick check: is a book already vectorized?
```python
from pathlib import Path
vector_path = Path("vectors") / "guerra_y_paz" / "1000" / "200"
exists_and_nonempty = vector_path.exists() and any(vector_path.iterdir())
print(exists_and_nonempty)
```
