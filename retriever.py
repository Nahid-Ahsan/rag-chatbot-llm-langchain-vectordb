# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


def get_text(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Filter out documents with None or empty content
    valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]

    if not valid_documents:
        raise ValueError("No valid documents found in the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    splits = text_splitter.split_documents(valid_documents)
    return splits


