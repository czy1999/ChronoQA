import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from FlagEmbedding import FlagModel
import time
import editdistance
from tqdm import tqdm
from datetime import datetime
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from prompt import EVENT_EXTRACT_PROMPT
from api import llm_stream
import json
from tinydb import TinyDB, Query
import concurrent.futures
import argparse
import re
import requests

class ChromaAPIServer:
    def __init__(self, path = None):
        if path is None:
            self.ip = 'http://localhost:8199'
        else:
            self.ip = path


    def request_chunks_multi(self,query,n_results=10,where=None,where_document=None):
        response = requests.post(f"{self.ip}/query_chunks_multi/", json={"query": query,"n_results":n_results,"where":where,"where_document":where_document})
        return response.json()

    def request_chunks(self,query,n_results=10,where=None,where_document=None):
        response = requests.post(f"{self.ip}/query_chunks/", json={"query": query,"n_results":n_results,"where":where,"where_document":where_document})
        return response.json()
    
api_client = ChromaAPIServer(path='http://localhost:8199')


def extract_dates(text):
    patterns = [
        (r'\b\d{4}年\d{1,2}月\d{1,2}日', '当天'),    # 'YYYY年M月D日'
        (r'\b\d{4}年\d{1,2}月', '当月'),          # 'YYYY年M月'
        (r'\b\d{4}年', '当年')                     # 'YYYY年'
    ]

    for pattern, date_type in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0), date_type

    return None,None

def extract_event_from_chunks(chunks, date_filter):
    prompt = f"{EVENT_EXTRACT_PROMPT.replace('{passages}', chunks)}"
    try:
        response = llm_stream(prompt, stream=False)
        result = eval(response)
        return result
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return None

def construct_composite_query(query,date_filter, event_name, time_type):
    time_expression = f"{event_name}{time_type}，"
    composite_query = query.replace(date_filter, time_expression).replace('，，', '，')
    return composite_query

def process_composite_query(query):
    date_filter, time_type = extract_dates(query)

    if not date_filter:
        # print("No valid date found in the query.")
        return None

    chunks = api_client.request_chunks(query='', where_document ={"$contains": date_filter}, n_results=20)
    chunks = '\n\n'.join([x[-1] for x in chunks])
    if not chunks:
        # print("No relevant chunks found for the given date filter.")
        return None

    result = extract_event_from_chunks(chunks, date_filter)
    if result and result.get('event_name'):
        composite_query = construct_composite_query(query,date_filter, result['event_name'], time_type)
        return (composite_query,result)
    else:
        # print("No event name found in the response.")
        return None
    
def process_composite_qa_json(qa_json):
    query = qa_json['question']
    composite_query = process_composite_query(query)
    if composite_query:
        composite_query,result = composite_query
        qa_json['original_question'] = query
        qa_json['question'] = composite_query
        qa_json['temporal_expression_type'] = 'implicit'
        qa_json['reference_document_count'] = 'multiple'
        qa_json['event_name'] = result['event_name']
        qa_json['event_time'] = result['time']
        del qa_json['thoughts']
        del qa_json['ref_prompt']
    else:
        return None
    return qa_json

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_composite_query_multi(qa_list):
    
    def process_qa(qa):
        chunks = qa['time_chunks']
        date_filter = qa['date_filter']
        time_type = qa['time_type']
        result = extract_event_from_chunks(chunks, date_filter)
        if result and result.get('event_name'):
            composite_query = construct_composite_query(qa['question'], date_filter, result['event_name'], time_type)
            qa['original_question'] = qa['question']
            qa['question'] = composite_query
            qa['temporal_expression_type'] = 'implicit'
            qa['reference_document_count'] = 'multiple'
            qa['event_name'] = result['event_name']
            qa['event_time'] = result['time']
            del qa['thoughts']
            del qa['ref_prompt']
            del qa['time_chunks']
            del qa['date_filter']
            del qa['time_type']
            return qa
        return None
    
    filtered_qa_list = []
    print('start filtering...')
    for qa in tqdm(qa_list):
        query = qa['question']
        date_filter, time_type = extract_dates(query)
        if date_filter:
            chunks = api_client.request_chunks(query='', where_document={"$contains": date_filter}, n_results=30)
            chunks = '\n\n'.join([x[-1] for x in chunks])
            if chunks:
                qa['date_filter'] = date_filter
                qa['time_type'] = time_type
                qa['time_chunks'] = chunks
                filtered_qa_list.append(qa)
    print(len(filtered_qa_list), 'filtered')
    
    print('start extracting...')
    processed_qas = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_qa = {executor.submit(process_qa, qa): qa for qa in filtered_qa_list}
        for future in tqdm(as_completed(future_to_qa), total=len(filtered_qa_list)):
            result = future.result()
            if result:
                processed_qas.append(result)
    return processed_qas

if __name__ == "__main__":
    # with open('/home/czy/code/temporal_rag/news_crawer/data/sina_news_19-24_with_response_split_v5.pkl','rb') as f:
    #     df = pickle.load(f)

    # print(df.columns)
    # print(df.head())

    seed_qa = pd.read_csv('/home/czy/code/temporal_rag/news_crawer/dataset/qa_dataset_single_qa.csv',index_col=0)
    seed_qa = seed_qa.to_dict(orient='records')
    for qa in seed_qa[55:]:
        query = qa['question']
        qa_json = process_composite_qa_json(qa)
        exit()

