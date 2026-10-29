import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from core.vector_store import build_vector_store, get_retriever, load_vector_store

load_dotenv()

SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        # Retry automatically on Groq rate-limit (429) errors
        max_retries=6,
    )


def format_docs(docs):
    """Join the retrieved chunks into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain(vector_store):
    """Full LCEL RAG pipeline: retrieve -> prompt -> LLM -> string."""
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    """Build a new vector store from the transcript and return the RAG chain."""
    vector_store = build_vector_store(transcript)
    return _build_chain(vector_store)


def load_rag_chain():
    """Return a RAG chain that uses the vector store already saved on disk."""
    vector_store = load_vector_store()
    return _build_chain(vector_store)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer