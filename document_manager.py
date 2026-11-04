
import os
import shutil

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_DIR = Path("data/pdfs")
VECTOR_DB_DIR = "vector_db"

PDF_DIR.mkdir(parents=True, exist_ok=True)


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def add_pdf_to_knowledge_base(pdf_file):

    if pdf_file is None:
        return "Please upload a PDF file."

    uploaded_path = Path(pdf_file)

    if uploaded_path.suffix.lower() != ".pdf":
        return "Only PDF files are supported."

    destination = PDF_DIR / uploaded_path.name

    if destination.exists():
        return "This PDF already exists in the knowledge base."

    shutil.copy2(uploaded_path, destination)

    try:
        loader = PyPDFLoader(str(destination))
        documents = loader.load()

        if not documents or not any(
            doc.page_content.strip() for doc in documents
        ):
            destination.unlink(missing_ok=True)
            return "No readable text found in this PDF."

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            destination.unlink(missing_ok=True)
            return "No text chunks could be created."

        if os.path.exists(VECTOR_DB_DIR):

            vector_store = FAISS.load_local(
                VECTOR_DB_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )

            vector_store.add_documents(chunks)

        else:

            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )

        # Save updated database
        vector_store.save_local(VECTOR_DB_DIR)

        return (
            f"Successfully added: {uploaded_path.name}\n"
            f"Pages processed: {len(documents)}\n"
            f"Chunks added: {len(chunks)}"
        )

    except Exception as e:

        if destination.exists():
            destination.unlink()

        return f"Error processing PDF: {str(e)}"
