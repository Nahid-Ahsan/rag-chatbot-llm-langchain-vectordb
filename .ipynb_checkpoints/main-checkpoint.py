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
import json 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
import time
import textwrap

import warnings
warnings.filterwarnings('ignore')

splits = get_text('/home/nahid/codes/rag/content/M-618.pdf')
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectordb = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectordb.as_retriever(search_kwargs = {"k": 2, "search_type" : "similarity"})


llm = HuggingFacePipeline(pipeline=query_pipeline)


prompt_template = """
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

Context: {context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)

def wrap_text_preserve_newlines(text, width=700):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    
    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4]
            + ' - page: '
            + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )
    
    # ans = ans + '\n\nSources: \n' + sources_used
    
    return ans

def extract_answer(response):
    # Find the start index of the "Answer:" section
    start_index = response.find("Answer:") + len("Answer:")
    
    # Slice the response starting from the "Answer:" section
    response_from_answer = response[start_index:]
    
    # Optional: Find the end index if there is a "Time elapsed" section
    end_index = response_from_answer.find("Time elapsed:")
    
    if end_index != -1:
        # Slice up to the "Time elapsed:" section if it exists
        answer = response_from_answer[:end_index].strip()
    else:
        # Otherwise, use the whole response_from_answer
        answer = response_from_answer.strip()
    
    return answer

def llm_ans(query):
    start = time.time()
    
    llm_response = qa_chain.invoke(query)
    ans = llm_response['result']
    print('type: ', type(ans))
    # ans = process_llm_response(llm_response)
    
    end = time.time()

    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    # return extract_answer(ans)
    return  ans + time_elapsed_str


query = "what is Health Insurance Marketplace?"
response = llm_ans(query)
print('response: ',response)


