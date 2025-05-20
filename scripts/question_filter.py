import pandas as pd
from prompt import VERIFY_PROMPT
from api import llm_stream
import concurrent.futures
from tqdm import tqdm

def filter_questions_multithreaded(seed_qa, output_path, max_workers=5):
    """
    Filters Q&A pairs using multi-threading and saves the filtered results to a CSV file.

    Args:
        seed_qa (list of dict): List of Q&A dictionaries to be filtered. Each dictionary should have 'question' and 'answer' keys.
        verify_prompt (str): The prompt template containing '{qa_pair}' placeholder.
        output_path (str): Path to save the filtered Q&A pairs as a CSV file.
        max_workers (int, optional): The maximum number of threads to use. Defaults to 5.

    Example:
        sub_queries_path = './dataset/qa_dataset_single_qa.csv'
        output_path = './dataset/qa_dataset_single_qa_filter.csv'
        sub_query_df = pd.read_csv(sub_queries_path, index_col=0)
        seed_qa = sub_query_df.to_dict(orient='records')
        
        filter_questions_multithreaded(seed_qa, VERIFY_PROMPT, output_path, max_workers=10, subset=100)
    """

    
    filter_qa = []

    def process_qa(qa):
        qa_pair = f"问题：{qa['question']} 答案：{qa['answer']}"
        res = llm_stream(VERIFY_PROMPT.replace('{qa_pair}', qa_pair), stream=False,model='gpt-4o-mini')
        try:
            score = eval(res).get('score', 0)
        except Exception:
            score = 0
        if score == 1:
            return qa
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_qa, qa): qa for qa in seed_qa}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing QAs"):
            result = future.result()
            if result is not None:
                filter_qa.append(result)

    filtered_df = pd.DataFrame(filter_qa)
    filtered_df.to_csv(output_path, index=False)
    print('Filter QA pairs num: ', len(filtered_df))
    print(f"Filtered QA pairs have been saved to {output_path}")
    return filtered_df

if __name__ == '__main__':
    # 读取原始 Q&A 数据
    # sub_queries_path = './dataset/qa_dataset_single_qa_v2_original.csv'
    # output_path = './dataset/qa_dataset_single_qa_v2_filter.csv'
    # sub_queries_path = './dataset/qa_dataset_multi_parallel_qa_v2_original.csv'
    # output_path = './dataset/qa_dataset_multi_parallel_qa_v2_filter.csv'
    sub_queries_path = '/home/czy/code/temporal_rag/news_crawer/dataset/TempRAG/qa.csv'
    output_path = '/home/czy/code/temporal_rag/news_crawer/dataset/TempRAG/qa_filter.csv'
    sub_query_df = pd.read_csv(sub_queries_path, index_col=0)
    seed_qa = sub_query_df.to_dict(orient='records')
    filter_questions_multithreaded(seed_qa, output_path, max_workers=50)