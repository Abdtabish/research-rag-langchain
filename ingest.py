from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


pdf_folder = Path("data/pdfs")

pdf_files = list(pdf_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF(s).")


documents = []

for pdf_file in pdf_files:

    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    pdf_documents = loader.load()

    documents.extend(pdf_documents)

print(f"\nTotal pages loaded: {len(documents)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded.")


vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created.")


vector_store.save_local("vector_db")

print("FAISS vector store saved.")
