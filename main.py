import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from generator import query_pipeline
from retriever import get_text
import transformers
from transformers import AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS

from IPython.display import display, Markdown
def colorize_text(text):
    for word, color in zip(["Reasoning", "Question", "Answer", "Total time"], ["blue", "red", "green", "magenta"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text

# RAG Setup
# Create Ollama embeddings and vector store
splits = get_text('/home/nahid/codes/rag/test.pdf')
model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"device": "cuda:1"}
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectordb = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectordb.as_retriever()
llm = HuggingFacePipeline(pipeline=query_pipeline)
# Prompt
# prompt = hub.pull("rlm/rag-prompt")
# Post-processing
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Question
# print(rag_chain.invoke("What is Task Decomposition?"))

# exit()
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

def rag(qa, query):
    response = qa.run(query)
    # full_response =  f"Question: {query}\nAnswer: {response}"
    # display(Markdown(colorize_text(full_response)))
    print(response)
    
query = "what architecture followed by Vision Transformer?"
rag(qa, query)