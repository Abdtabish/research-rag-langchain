# Research Paper RAG Application

An AI-powered Research Paper Question Answering Application built using **LangChain, Hugging Face Embeddings, FAISS, Groq LLM, and Gradio**.

## Overview

This application allows users to upload research papers in PDF format and ask questions about their content.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks and generate context-grounded answers with source attribution.

## Tech Stack

- Python
- LangChain
- Hugging Face Embeddings (BAAI/bge-small-en-v1.5)
- FAISS Vector Database
- Groq LLM
- Gradio
- PyPDF

## Features

- Multiple PDF document ingestion
- PDF text extraction
- Recursive text splitting with overlapping chunks
- Semantic embeddings using Hugging Face
- FAISS-based similarity search
- Retrieval-Augmented Generation
- Context-grounded answers
- Source document and page attribution
- Interactive Gradio interface

## Project Architecture

1. Load PDF documents.
2. Extract text from PDFs.
3. Split documents into smaller chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks based on user questions.
7. Pass retrieved context to the Groq LLM.
8. Generate answers and display source references.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd research-rag-langchain
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file.

### 5. Add PDF documents

Place your research PDFs inside:

```text
data/pdfs/
```

### 6. Build the vector database

```bash
python ingest.py
```

### 7. Run the application

```bash
python app.py
```

## Performance

Baseline performance observed during development:

| Metric | Time |
|---|---:|
| Retrieval | 0.18 seconds |
| LLM Generation | 43.67 seconds |
| Total | 43.85 seconds |

*These are development measurements and may vary depending on the model, network, and execution environment.*

## Future Improvements

- RAG evaluation using Faithfulness and Answer Relevancy
- Improved retrieval strategies
- Response streaming
- Conversation history
- Performance optimization

## Author

Abdul Tabish
