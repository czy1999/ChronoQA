import random
import pandas as pd
import pickle
from api import gte_embedding
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
from prompt import GENERATE_QUESTION_PROMPT, COMPOSITE_QUESTION, QA_TYPES
from api import llm_stream
import json
from tinydb import TinyDB, Query
import concurrent.futures
import argparse
from composite_question import process_composite_query,process_composite_qa_json,api_client,process_composite_query_multi

def get_embedding_model():
    model = FlagModel('BAAI/bge-large-zh-v1.5', use_fp16=True)
    return model

def get_passages():
    with open('./data/sina_news_19-24_with_response_split_v5.pkl','rb') as f:
        df = pickle.load(f)
    return df['summary']  

def get_precomputed_embeddings():
    with open('./data/summary_embedding_bge.pkl','rb') as f:
        embeddings = pickle.load(f)
    print('embeddings loaded',embeddings.shape)
    return embeddings

def query_passages(query,passages,model = None,embeddings=None ,k=3,method='embedding'):
    if method == 'embedding':
        if model is None:
            raise ValueError('Embedding model is not provided')
        query_embedding = model.encode_queries(query)
        if embeddings is None:
            embeddings = model.encode(passages)
        similarities = query_embedding@embeddings.T
        top_k_indices = np.argsort(-similarities)[:k]
        top_k_passages = [passages[i] for i in top_k_indices]
        # top_k_passages = [{'similarity':similarities[i],'passage':passages[i]} for i in top_k_indices]
    elif method == 'edit_distance':
        distances = [(i, editdistance.distance(query, passage)) for i, passage in enumerate(passages)]
        distances.sort(key=lambda x: x[1])
        top_k_passages =  [passages[i] for i, _ in distances[:k]]
    else:
        raise ValueError('method not supported')
    return top_k_passages


def get_meta_data_list(df):
    def get_meta_data(x):
        meta_data = {}
        meta_data['title'] = x['title']
        meta_data['date'] = x['date']
        meta_data['date_num'] = int(x['date'].replace('-',''))
        return meta_data
    df['meta_data'] = df.apply(lambda x:get_meta_data(x),axis=1)
    return df['meta_data'].to_list()

def create_collection(df,db_path = './vector_db',collection_name = 'test'):
    bge = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-zh-v1.5')
    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection(name=collection_name, embedding_function=bge, metadata={"hnsw:space": "ip"})
    uri_ids = list(df.index)
    batch_size = 30000  
    total_data = len(uri_ids)
    passages = df['summary'].apply(str).to_list()
    embeddings = get_precomputed_embeddings()
    meta_data_list = get_meta_data_list(df)
    for i in range(0, total_data, batch_size):
        batch_ids = [str(x) for x in uri_ids[i:i + batch_size]]
        batch_passages = passages[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_meta_data = meta_data_list[i:i + batch_size]
        collection.add(ids=batch_ids, documents=batch_passages, embeddings=batch_embeddings,metadatas=batch_meta_data)
        print(f"Inserted batch {i // batch_size + 1}")
    print('collection created')
    return collection

def get_collection(db_path = './vector_db',collection_name = 'test',from_web=False):
    bge = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-zh-v1.5')
    if from_web:
        chroma_client = chromadb.HttpClient(host='localhost', port=8000)
        collection = chroma_client.get_collection(collection_name)
    else:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=collection_name, embedding_function=bge, metadata={"hnsw:space": "ip"})
    return collection

def get_db(db_path = './dataset/qa_dataset.json',table_name = 'qa'):
    db = TinyDB(db_path)
    table = db.table(table_name)
    return table

def get_query_chunks(query,collection,n_results = 10):
    results = collection.query(
        query_texts=query,
        n_results=n_results,
        # where={"date_num": {"$gt": 20240101}}, # optional filter
    )

    chunks = ''
    for i in range(n_results):
        chunks += results['documents'][0][i] + '\n'
        chunks += 'Published Time:' + results['metadatas'][0][i]['date'] + '\n\n'
    return chunks

def generate_question_json(query,chunks,type_to_generate,current_date = None):
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
    if type_to_generate is None:
        type_to_generate = 'precise_direct_expression'
    if type_to_generate.startswith('multi'):
        prompt = COMPOSITE_QUESTION.replace('{qa_type_instruction}',QA_TYPES[type_to_generate]).replace('{current_date}',current_date).replace('{chunks}',chunks)
    else:
        prompt = GENERATE_QUESTION_PROMPT.replace('{qa_type_instruction}',QA_TYPES[type_to_generate]).replace('{current_date}',current_date).replace('{chunks}',chunks)
    try:
        res = llm_stream(prompt,stream=False)
        res = res.replace('```json','').replace('```','')
        res = eval(res)
        res['date'] = current_date
        res['source'] = query
        res['ref_chunks'] = chunks
        res['ref_prompt'] = type_to_generate
    except Exception as e:
        print(e)
        return None
    if res['can_generate'] == 0:
        return None
    return res

def generate_questions_for_seed_articles(df,seed_articles_path = './dataset/seed_articles.csv'):

    c = df[df['date']>'2024'].sample(frac = 0.1)
    a = df[df['date']>'2023']
    b = a[a['date']<'2024'].sample(frac = 0.05)
    d = df.sample(frac = 0.01)
    df = pd.concat([b,c,d]).drop_duplicates(subset=['url']).copy()
    # queries = df.summary.to_list()
    # collection = get_collection(collection_name='news')
    # # query = collection.get(ids = ['200000'])['documents'][0]
    # chunks_list = []
    # for query in tqdm(queries):
    #     chunks = get_query_chunks(query,collection=collection)
    #     chunks_list.append(chunks)
    # df['chunks'] = chunks_list
    df[['summary','date']].to_csv(seed_articles_path)
    print('done')

if __name__ == "__main__":
    # with open('/home/czy/code/temporal_rag/news_crawer/data/sina_news_19-24_with_response_split_v5.pkl','rb') as f:
    #     df = pickle.load(f)

    # 加入参数设置
    args = argparse.ArgumentParser()
    args.add_argument('--seed_articles_path', default='./dataset/seed_articles_v2.csv')
    args.add_argument('--qa_dataset_path', default='./dataset/qa_dataset.json')
    args.add_argument('--qa_dataset_table', default='qa')
    args.add_argument('--save_to', default='./dataset/qa_dataset.csv')
    args.add_argument('--max_workers', default=50)
    args.add_argument('--generate_single_direct_expression', default=True)
    args.add_argument('--generate_single_relative_expression', default=True)
    args.add_argument('--generate_multi_ordinal_expression', default=True)
    args.add_argument('--generate_multi_judgement_expression', default=True)
    args.add_argument('--generate_multi_direct_expression', default=True)
    args.add_argument('--generate_multi_composite_expression', default=True)
    args.add_argument('--no_chunks', default=False)
    args.add_argument('--method', default='embedding')
    args.add_argument('--sub_queries_path', default= '')
    args.add_argument('--max_per_qa_num', default=1000)
    args = args.parse_args()

    args.qa_dataset_path = './dataset/TempRAG/qa.json'
    args.qa_dataset_table = 'test'
    args.save_to = './dataset/TempRAG/qa.csv'
    args.seed_articles_path = './dataset/seed_articles/seed_articles_large.csv'
    args.sub_queries_path = './dataset/seed_single_qa/seed_single_qa_filter.csv'
    args.max_per_qa_num = 2000
    args.generate_single_direct_expression = False
    args.generate_single_relative_expression = True
    args.generate_multi_ordinal_expression = True
    args.generate_multi_judgement_expression = True
    args.generate_multi_direct_expression = True
    args.generate_multi_composite_expression = True

    args.no_chunks = True
    args.method = 'embedding'
    table = get_db(db_path = args.qa_dataset_path,table_name=args.qa_dataset_table)

    # Read seed articles for single qa generation
    seed_passage_df = pd.read_csv(args.seed_articles_path)
    # 先shuffle
    queries = seed_passage_df.sample(frac=1).summary.to_list()[:args.max_per_qa_num]
    if args.no_chunks:
        chunks = queries
    else:
        chunks = seed_passage_df.chunks.to_list()
    
    # Read sub queries for multi qa generation
    if args.sub_queries_path:
        sub_query_df = pd.read_csv(args.sub_queries_path,index_col=0)
        sub_query_df = sub_query_df[sub_query_df['temporal_type']=='direct']
        sub_qa_lists = sub_query_df.apply(lambda x:'问题：' + x.question + '答案：'+x.answer,axis = 1).to_list()

        if any([
            args.generate_multi_direct_expression,
            args.generate_multi_ordinal_expression,
            args.generate_multi_judgement_expression
        ]):
            if args.method == 'embedding':  
                model = get_embedding_model()
                embeddings = model.encode(sub_qa_lists)
                print('Using embedding method to query sub qa pairs')
            else:
                model = None
                embeddings = None
                print('Using edit distance method to query sub qa pairs')
            random.shuffle(sub_qa_lists)
            multi_queries = []
            sim_query_list = []
            print('start generating similar sub qa pairs...')
            for i in tqdm(sub_qa_lists[:args.max_per_qa_num]):
                sim_queries = query_passages(i,sub_qa_lists,model=model,embeddings=embeddings,method=args.method,k=20)
                sim_queries_str = '\n'.join(sim_queries)
                multi_queries.append(i)
                sim_query_list.append(sim_queries_str)
    else:
        print('no sub queries provided,skip multi qa generation')


    def process_query(query, chunk, type_to_generate,model = None):
        return generate_question_json(query, chunk, type_to_generate,model)

    # 加入tqdm 
    def process_queries_in_parallel(queries, chunks, type_to_generate,model = None, max_workers=100):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_query, query, chunk, type_to_generate,model) for query, chunk in zip(queries, chunks)]
            results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures),total=len(queries))]
        return results
    
    if args.generate_single_direct_expression:
        print('type:single_direct_expression,start generating questions...')
        results = process_queries_in_parallel(queries, chunks, type_to_generate='single_direct_expression')
        res = [x for x in results if x is not None]
        print('inserting ',len(res))
        table.insert_multiple(res)            
        print('done')

    if args.generate_single_relative_expression:
        print('type:single_relative_expression,start generating questions...')
        results = process_queries_in_parallel(queries, chunks = chunks, type_to_generate='single_relative_expression')
        res = [x for x in results if x is not None]
        print('inserting ',len(res))
        table.insert_multiple(res)            
        print('done')

    if args.generate_multi_ordinal_expression:
        print('type:multi_ordinal_expression,start generating questions...')
        results = process_queries_in_parallel(multi_queries, chunks = sim_query_list, type_to_generate='multi_ordinal_expression',model='o1-mini',max_workers=50)
        res = [x for x in results if x is not None]
        print('inserting ',len(res))
        table.insert_multiple(res)
        print('done')

    if args.generate_multi_judgement_expression:
        print('type:multi_judgement_expression,start generating questions...')
        results = process_queries_in_parallel(multi_queries, chunks = sim_query_list, type_to_generate='multi_judgement_expression',model='o1-mini',max_workers=50)
        res = [x for x in results if x is not None]
        print('inserting ',len(res))
        table.insert_multiple(res)
        print('done')

    if args.generate_multi_direct_expression:
        print('type:multi_direct_expression,start generating questions...')
        results = process_queries_in_parallel(multi_queries, chunks = sim_query_list, type_to_generate='multi_direct_expression',model='o1-mini',max_workers=50)
        res = [x for x in results if x is not None]
        print('inserting ',len(res))
        table.insert_multiple(res)            
        print('done')

    if args.generate_multi_composite_expression:
        seed_qa = sub_query_df.sample(frac=1).to_dict(orient='records')[:args.max_per_qa_num]
        results = []
        print('start generating composite questions...')
        res = process_composite_query_multi(seed_qa)
        print('inserting ',len(res))
        table.insert_multiple(res)            
        print('done')
    
    df = pd.DataFrame(table.all())
    df.to_csv(args.save_to)
    print('done,file saved to ',args.save_to)
    # with open('./dataset/qa_dataset.json','wb') as f:
    #     data = json.load(f)