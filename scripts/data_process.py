import pandas as pd
from datetime import datetime
from  tinydb import TinyDB
import re
import json
import os
import dask.dataframe as dd
import time
import concurrent.futures
import numpy as np
import pickle
from prompt import SUMMARY_PROMPT
from tqdm import tqdm
from zhipuai import ZhipuAI
import logging
from api import gte_embedding

API_KEY = os.getenv("ZHIPU_API_KEY")

def concat_data(folder_path):
    files = [x for x in os.listdir(folder_path) if x.endswith('.json')]
    dfs = []
    for file in files:
        file_path = os.path.join(folder_path,file)
        db = TinyDB(file_path, encoding='utf-8')
        table = db.table('news')
        data = table.all()
        current_df = pd.DataFrame(data)
        dfs.append(current_df)
    df = pd.concat(dfs,ignore_index=True)
    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    del df['comments']
    del df['author']
    del df['description']
    del df['sitename']
    del df['categories']
    del df['image']
    return df

def generate_prompt(item):
    res_list = []
    article = ''
    article += '新闻标题：' + item['title']
    article += '\n新闻发布日期：'+item['date']
    article += '\n新闻正文：'+item['text']
    prompt = SUMMARY_PROMPT.replace('{query}',article)
    return prompt

def generate_post(prompt,batch_name,uid):
    post = {
        "custom_id": batch_name+'_'+str(uid), #每个请求必须包含custom_id且是唯一的，用来将结果和输入进行匹配
        "method": "POST",
        "url": "/v4/chat/completions", 
        "body": {
            "model": "glm-4-flash", #每个batch文件只能包含对单个模型的请求,支持 glm-4-0520 glm-4-air、glm-4-flash、glm-4、glm-3-turbo.
            "messages": [
                {"role": "system","content": ""},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
    }
    return post

def generate_batch_post(df,batch_name):
    res_list = []
    for uid in tqdm(df.index):
        item = df.loc[uid]
        prompt = generate_prompt(item)
        post = generate_post(prompt,batch_name,uid)
        res_list.append(post.copy())
    return res_list

def save_batch_post(res_list,batch_name,folder_path = './data/batch_data/'):
    with open(folder_path+batch_name+'.jsonl', 'w', encoding='utf-8') as f:
        for item in res_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 每五万条数据生成一个batch文件，写成一个函数
def split_df_to_batch_files(df,batch_size = 50000,folder_path = './data/batch_data/'):
    batch_num = len(df) // batch_size + 1
    for i in range(batch_num):
        batch_df = df[i*batch_size:(i+1)*batch_size]
        batch_name = 'batch_'+batch_df['date'].iloc[0]+'_'+batch_df['date'].iloc[-1]
        res_list = generate_batch_post(batch_df,batch_name)
        save_batch_post(res_list,batch_name,folder_path)
        print(batch_name+' saved')

def logging_info_to_file(info,file_path = './log/'):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    log_file = os.path.join(file_path, f'create_batch_task_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)
    logging.info(info)


def create_batch_task_from_files(batch_data_path = './data/batch_data/',api_key=API_KEY):
    files = [x for x in os.listdir(batch_data_path) if x.endswith('.jsonl')]
    task_id_list = []
    for filename in files:
        logging_info_to_file('start to create file,filename:%s' + filename)
        client = ZhipuAI(api_key=api_key) # 请填写您自己的APIKey
        result = client.files.create(
            file=open(batch_data_path+filename,"rb"),
            purpose="batch"
        )
        logging_info_to_file('file created,file_id:%s' + str(result.id))
        create = client.batches.create(
            input_file_id=result.id,
            endpoint="/v4/chat/completions", 
            completion_window="24h", #完成时间只支持 24 小时
            metadata={
                "description": filename
            }
        )
        logging_info_to_file('task created,task_id:%s' + str(create.id))
        task_id_list.append(create.id)
    with open(batch_data_path +'task_id_list_{}.pkl'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),'wb') as f:
        pickle.dump(task_id_list,f)
    return task_id_list

def get_task_status(task_id_list,api_key=API_KEY):
    client = ZhipuAI(api_key=api_key) # 请填写您自己的APIKey
    total = 0
    succeed = 0
    failed = 0
    succeed_list = []
    failed_list = []
    for task_id in tqdm(task_id_list,desc='Get task status',total=len(task_id_list)):
        status = client.batches.retrieve(task_id).status
        if status == 'completed':
            succeed += 1
            succeed_list.append(task_id)
        else:
            failed += 1
            failed_list.append(task_id)
        total += 1
    print('Failed task_id: %s' % failed_list)
    print('Total: %d, Succeed: %d, InProgress: %d' % (total, succeed, failed))
    return succeed_list,failed_list

def get_task_result(task_id_list,output_file_path = './data/batch_data/output_file/',api_key=API_KEY):
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
        print('output_file_path %s created' % output_file_path)
    client = ZhipuAI(api_key=api_key) # 请填写您自己的APIKey
    for task_id in task_id_list:
        result = client.batches.retrieve(task_id).output_file_id
        file_name = client.batches.retrieve(task_id).metadata['description']
        content = client.files.content(result) 
        content.write_to_file(output_file_path+'output_'+file_name)
        print('output_'+file_name+' saved')

def get_task_status_and_result(task_id_list,output_file_path = './data/batch_data/output_file/',api_key=API_KEY):
    client = ZhipuAI(api_key=api_key) # 请填写您自己的APIKey
    for task_id in task_id_list:
        status = client.batches.retrieve(task_id).status
        print(task_id,status)
        if status == 'completed':
            result = client.batches.retrieve(task_id).output_file_id    
            file_name = client.batches.retrieve(task_id).metadata.description
            content = client.files.content(result) 
            content.write_to_file(output_file_path+'output_'+file_name)
            print('output_'+file_name+' saved')
        else:
            print('task_id:%s is not SUCCEED' % task_id)


def extract_json_content(text):
    # 使用正则表达式匹配 ```json\n 和 ``` 之间的内容
    pattern = r"```json\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def process_output_file_to_csv(batch_data_path = './data/batch_data/output_file/',output_file = None):
    if output_file is None:
        output_file = batch_data_path + 'all_response.csv'
    files = [x for x in os.listdir(batch_data_path) if x.endswith('.jsonl')]
    json_list = []
    print('Start to process output file')
    for file in files:
        with open(batch_data_path+file,'r',encoding='utf-8') as f:
            for line in f:
                content = json.loads(line)
                json_list.append(content)
    print('Total response:',len(json_list))
    
    response_list = []
    for item in tqdm(json_list,desc='Extract json content'):
        request_id = int(item['response']['body']['request_id'].split('_')[-1])
        res_text = item['response']['body']['choices'][0]['message']['content']
        try:
            res_dict = json.loads(extract_json_content(res_text)[0])
            response_list.append({'request_id':request_id,'response':res_dict})
        except:
            continue
        
    df_response = pd.DataFrame(response_list)
    df_response.to_csv(output_file)
    print('Response saved at %s' % output_file)

def get_failed_df(df,response,output_folder = './data/batch_data/output_file/'):
    df_drop = df[~df.index.isin(response.request_id)]
    to_generate_prompt_list = []
    for index,row in df_drop.iterrows():
        prompt = generate_prompt(row)
        to_generate_prompt_list.append({'index':index,'prompt':prompt})
    
    with open(output_folder+'to_generate_prompt_list.pkl','wb') as f:
        pickle.dump(to_generate_prompt_list,f)
    df_drop.to_csv(output_folder+'failed_df.csv')
    print('Failed df saved at %s' % output_folder+'failed_df.csv')
    print('To generate prompt list saved at %s' % output_folder+'to_generate_prompt_list.pkl')
    return df_drop

def merge_df_and_response(df,df_response):
    df.reset_index(inplace=True)
    df_response.rename(columns={'request_id':'index'},inplace=True)
    df_response['index'] = df_response['index'].astype(int)
    df = df.merge(df_response,on='index')
    df['response'] = df['response'].apply(lambda x: eval(x))
    df.drop(columns=['index'],inplace=True)
    return df

def split_response_to_rewrite_text_and_related_kg(df):
    def get_rewrite_text_from_response(x):
        if type(x) is not dict:
            return None
        if 'rewrite_text' in x.keys():
            return x['rewrite_text']
        else:
            return None
    def get_related_kg_from_response(x):
        if type(x) is not dict:
            return None
        if 'related_kg' in x.keys():
            return x['related_kg']
        else:
            return None
    df['summary'] = df['response'].apply(lambda x: get_rewrite_text_from_response(x))
    df['kgs'] = df['response'].apply(lambda x: get_related_kg_from_response(x))
    df.dropna(inplace=True)
    return df


def split_list(data_list, num_splits):
    """
    将一个列表划分为指定数量的子集，确保有序且每个子集的大小尽可能均匀。
    """
    # 计算每个子集的基础大小
    subset_size = len(data_list) // num_splits
    # 计算多余的元素个数
    remainder = len(data_list) % num_splits

    subsets = []
    start_idx = 0

    for i in range(num_splits):
        # 确定每个子集的结束索引
        end_idx = start_idx + subset_size + (1 if i < remainder else 0)
        subsets.append(data_list[start_idx:end_idx])
        start_idx = end_idx

    return subsets

if __name__ == "__main__":
    with open('./data/sina_news_19-24_with_response_split_v5.pkl','rb') as f:
        df = pickle.load(f)



