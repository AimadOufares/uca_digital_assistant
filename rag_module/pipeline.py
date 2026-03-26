from .ingestion import fetch_documents
from .processing import process_documents
from .indexing import index_documents
from .rag_engine import answer_question

def run_pipeline(url, question):
    docs = fetch_documents(url)
    processed = process_documents(docs)
    index_documents(processed)
    return answer_question(question)