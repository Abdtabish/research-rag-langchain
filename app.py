
import gradio as gr
from document_manager import add_pdf_to_knowledge_base
from rag import rag_chain
import time
from rag import retriever, prompt, llm, format_docs
from langchain_core.output_parsers import StrOutputParser


def chat_with_rag(message, history):

    start = time.perf_counter()

    documents = retriever.invoke(message)

    retrieval_time = time.perf_counter() - start

    context = format_docs(documents)

    start = time.perf_counter()

    answer = (
        prompt | llm | StrOutputParser()
    ).invoke({
        "context": context,
        "question": message
    })

    llm_time = time.perf_counter() - start

    sources = []

    for doc in documents:

        page = doc.metadata.get("page", "Unknown")
        filename = doc.metadata.get("source", "Unknown")

        if isinstance(page, int):
            page += 1

        source = f"- {filename} | Page: {page}"

        if source not in sources:
            sources.append(source)

    final_response = answer

    if sources:
        final_response += "\n\n**Sources:**\n"
        final_response += "\n".join(sources)

    print("\n------ PERFORMANCE REPORT ------")
    print(f"Retrieval time: {retrieval_time:.2f} seconds")
    print(f"LLM generation time: {llm_time:.2f} seconds")
    print(f"Total time: {retrieval_time + llm_time:.2f} seconds")
    print("--------------------------------\n")

    return final_response


def upload_and_process_pdf(pdf_file):

    result = add_pdf_to_knowledge_base(pdf_file)

    return result


with gr.Blocks() as app:

    gr.Markdown("# Research Paper RAG Assistant")

    with gr.Tab("Chat"):

        gr.ChatInterface(
            fn=chat_with_rag,
            title="Ask Questions About Your Documents",
            description="Ask questions based on your uploaded research papers."
        )

    with gr.Tab("Manage Documents"):

        gr.Markdown("## Upload a New PDF")

        pdf_upload = gr.File(
            label="Choose PDF",
            file_types=[".pdf"],
            type="filepath"
        )

        upload_button = gr.Button(
            "Upload and Add to Knowledge Base"
        )

        upload_status = gr.Textbox(
            label="Upload Status",
            interactive=False
        )

        upload_button.click(
            fn=upload_and_process_pdf,
            inputs=pdf_upload,
            outputs=upload_status
        )

app.launch()
