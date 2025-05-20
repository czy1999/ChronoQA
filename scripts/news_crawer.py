from trafilatura import fetch_url,extract,bare_extraction
from tinydb import TinyDB, Query
import re
import threading
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import requests
from datetime import datetime
import time

def get_news_text(url):
    document = fetch_url(url)
    # text = extract(document)
    try:
        res_dict = bare_extraction(document, only_with_metadata=True)
    except Exception as e:
        print(e)
        print(url)
        return {}
    return res_dict

def extract_unique_urls_sina(text):
    pattern = r'https://[a-zA-Z0-9.-]+\.sina\.com\.cn/[a-zA-Z0-9/_-]+/doc-[a-zA-Z0-9]+\.shtml'
    urls = re.findall(pattern, text)
    unique_urls = list(set(urls))
    return unique_urls

def get_urls_from_sina(date):
    url = 'https://news.sina.com.cn/head/news{date}am.shtml'
    url = url.format(date=date.replace('-',''))
    html = requests.get(url)
    links = extract_unique_urls_sina(html.text)
    return links

def get_urls_from_sina_by_date(table_name = 'sina_urls',start_date = datetime(2019, 1, 1),end_date = datetime(2024, 8, 1)):
    table = get_db(table_name)
    date_range = pd.date_range(start=start_date, end=end_date)  
    dates = [date.strftime('%Y-%m-%d') for date in date_range]
    for date in tqdm(dates,desc='crawling'):      
        res = get_urls_from_sina(date)
        current_dict = {}
        current_dict['date'] = date
        current_dict['urls'] = res
        table.insert(current_dict)

def get_db(name = 'news'):
    if name == 'sina_urls':
        db = TinyDB('./db.json', encoding='utf-8')
        table = db.table('sina_urls')
        return table
    else:
        db = TinyDB('./data/{name}.json'.format(name=name), encoding='utf-8')
        table = db.table('news')
        return table

# 使用re从url抽取日期信息
def get_date_from_url(url):
    pattern = r'/(\d{4}-\d{2}-\d{2})/doc-'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None
def insert_news(url,table,lock):
    res_dict = get_news_text(url)
    try:
        res_dict.pop('commentsbody')
        res_dict.pop('body')
        res_dict.pop('fingerprint')
        res_dict.pop('id')
        res_dict.pop('hostname')
        res_dict.pop('license')
        res_dict.pop('raw_text')
        res_dict.pop('language')
        res_dict.pop('pagetype')
    except Exception as e:
        return None
    # with lock:  # 确保只有一个线程能够获取锁进行写入
    #     table.insert(res_dict)
    return res_dict

def insert_news_thread(urls,table,max_workers = 5):
    lock = threading.Lock()
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(insert_news, url, table,lock) for url in urls]
        for future in tqdm(concurrent.futures.as_completed(futures),total=len(urls)):
            result = future.result()
            if result:
                all_results.append(result)
    with lock:
        if all_results:  # 确保有数据才插入
            table.insert_multiple(all_results)

if __name__ == "__main__":
    # # load urls from db
    # urls_table = get_db('sina_urls')
    # data = urls_table.all()
    # df = pd.DataFrame(data).set_index('date')

    # use dbname to crawl news from sina between start_date and end_date
    dbname = '20240801-20240802'
    table = get_db(dbname)
    lock = threading.Lock()
    start_date = datetime.strptime(dbname.split('-')[0],'%Y%m%d')
    end_date = datetime.strptime(dbname.split('-')[1],'%Y%m%d')
    date_range = pd.date_range(start= start_date,end = end_date) 
    dates = [date.strftime('%Y-%m-%d') for date in date_range]
    
    for date in dates:
        print('fetching date:',date)
        urls = get_urls_from_sina(date)
        insert_news_thread(urls,table,max_workers=10)
        print('fetching date done!:',date)


    # for date in dates:
    #     urls = get_urls_from_sina(date)
    #     print(urls)

    # table = get_db()
    # table.truncate()
    
    # urls = ['https://news.sina.com.cn/c/xl/2022-08-26/doc-imizirav9750903.shtml',
    #         'https://news.sina.com.cn/c/2024-08-01/doc-inchavuh6342848.shtml',
    #         'https://news.sina.com.cn/w/2024-08-28/doc-incmcvru2118285.shtml'
    #         ]
    # insert_news_thread(urls,table)
    # data = table.all()
    # df = pd.DataFrame(data)
    # print(df)

