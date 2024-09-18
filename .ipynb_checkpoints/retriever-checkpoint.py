from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import nltk 
nltk.download('punkt_tab')

def get_text(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]

    if not valid_documents:
        raise ValueError("No valid documents found in the PDF.")

    text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=200)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     separator="\n\n", 
#     chunk_size=1200, 
#     chunk_overlap=100, 
#     is_separator_regex=False,
#     model_name='text-embedding-3-small',
#     encoding_name='text-embedding-3-small', 
# )
    
    splits = text_splitter.split_documents(valid_documents)
    return splits


