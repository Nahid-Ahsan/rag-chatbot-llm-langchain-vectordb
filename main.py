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
# from langchain.chains import RetrievalQA
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS
import json 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
import time
import textwrap
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
import re
import warnings
warnings.filterwarnings('ignore')


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


splits = get_text('/home/nahid/codes/rag/content/test.pdf')
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectordb = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectordb.as_retriever(search_kwargs = {"k": 2, "search_type" : "similarity"})


llm = HuggingFacePipeline(pipeline=query_pipeline)

# RAG-Fusion
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion


# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)
# question = "please explain the vision transformer architecture"

def llm_response(question):

    result = final_rag_chain.invoke({"question":question})
    answer = re.search(r"Answer:\s*(.*)", result, re.DOTALL)
    if answer:
        answer = answer.group(1).strip()
    answer = re.search(r"(.*)(?=\nQuestion:)", answer, re.DOTALL)
    if answer:
        answer = answer.group(1).strip()
    return answer

