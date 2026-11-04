
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env file. "
        "Please configure your API key."
    )


try:

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("Embedding model loaded successfully.")

except Exception as e:

    raise RuntimeError(
        f"Failed to load embedding model: {e}"
    ) from e

try:

    if not os.path.exists("vector_db"):
        raise FileNotFoundError(
            "FAISS vector database folder not found."
        )

    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS vector database loaded successfully.")

except Exception as e:

    raise RuntimeError(
        f"Failed to load FAISS database: {e}"
    ) from e

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

try:

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        reasoning_effort="low"
    )

    print("Groq LLM initialized successfully.")

except Exception as e:

    raise RuntimeError(
        f"Failed to initialize Groq LLM: {e}"
    ) from e


prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful research paper assistant.

    Answer the user's question using ONLY the provided context.

    If the answer is not available in the context,
    respond with:
    "I don't know based on the provided documents."

    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

def format_docs(documents):

    return "\n\n".join(
        doc.page_content for doc in documents
    )

def retrieve_documents(question):

    # Validate question
    if not isinstance(question, str) or not question.strip():

        raise ValueError(
            "Question cannot be empty. "
            "Please enter a valid question."
        )

    try:

        documents = retriever.invoke(question)

        # Check retrieved documents
        if not documents:

            raise ValueError(
                "No relevant documents were retrieved."
            )

        return documents

    except ValueError:
        raise

    except Exception as e:

        raise RuntimeError(
            f"Document retrieval failed: {e}"
        ) from e


def generate_answer(inputs):

    try:

        response = (prompt | llm | StrOutputParser()).invoke(
            {
                "context": inputs["context"],
                "question": inputs["question"]
            }
        )

        if not response or not response.strip():

            raise ValueError(
                "LLM returned an empty response."
            )

        return response

    except Exception as e:

        raise RuntimeError(
            f"Answer generation failed: {e}"
        ) from e

rag_chain = (

    RunnableParallel(
        question=RunnablePassthrough(),
        documents=RunnableLambda(retrieve_documents)
    )

    |

    RunnableParallel(
        question=lambda x: x["question"],
        documents=lambda x: x["documents"],
        context=lambda x: format_docs(x["documents"])
    )

    |

    RunnableParallel(
        answer=RunnableLambda(generate_answer),
        documents=RunnableLambda(
            lambda x: x["documents"]
        )
    )

)

print("RAG pipeline initialized successfully.")
