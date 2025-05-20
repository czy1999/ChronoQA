import concurrent
import itertools
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import Sequence, List, Iterator, Tuple, Optional, Any, Callable, TypeVar
from openai import OpenAI
import requests
from tqdm import tqdm
import pickle
import re
import dashscope
import numpy as np
from dotenv import load_dotenv

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE")
GTE_API_KEY = os.getenv("GTE_API_KEY")

def assert_input_type(*types):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) != len(types):
                raise TypeError(f"{func.__name__} expects {len(types)} arguments, but got {len(args)}")
            for a, t in zip(args, types):
                if not isinstance(a, t):
                    raise TypeError(f"Argument {a} does not match expected type {t}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class _BaseApiConfig:
    def __repr__(self):
        return json.dumps(self.__dict__)

    def dump_to_json(self, json_path="./config.json"):
        if os.path.exists(json_path):  # avoid overwrite
            raise FileExistsError
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"Configuration dumped to {json_path}")

    @staticmethod
    def from_json_file(json_file) -> 'ApiConfig':
        with open(json_file, 'r') as f:
            data = json.load(f)
        return ApiConfig(**data)


@assert_input_type(str, str, str, int, int, float)
def _make_data(model_type, prompt, system_prompt, max_tokens, n, temperature):
    data = {
        'model': model_type,
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        'max_tokens': max_tokens,
        'n': n,
        'stop': None,
        'temperature': temperature
    }
    return data


def _make_header(api_key):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    return headers


class ApiConfig(_BaseApiConfig):
    def __init__(self, *, model_type: str, api_url: str, api_keys: Sequence[str]):
        self.model_type = model_type
        self.api_url = api_url
        self.api_keys = api_keys


def generate_response(api_key, model_type, api_url, prompt: Any, system_prompt="You are a helpful assistant.",
                      max_tokens=150, temperature=0.7, n=1, key=None):
    # 如果有key函数，应用key函数处理prompt
    #prompt: str = prompt if key is None else key(prompt)
    prompt = prompt['prompt']
    client = OpenAI(
        api_key=api_key,
        base_url=api_url
    ) 
    response = client.chat.completions.create(
        model=model_type,  
        messages=[    
            {"role": "system", "content": system_prompt},    
            {"role": "user", "content": prompt} 
        ],
        top_p=0.7,
        temperature=temperature
    ) 
    # 检查响应并返回内容
    result = response.choices[0].message.content.strip()
    # 避免API过载
    # time.sleep(0.5)
    
    return result


def process_prompts(api_config: ApiConfig, prompts: List[Any], **kwargs) -> Iterator[Tuple[str, Optional[str]]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_config.api_keys)*3) as executor:
        # Use itertools.cycle to cycle through api_keys
        api_key_cycle = itertools.cycle(api_config.api_keys)
        future_to_prompt = {
            executor.submit(partial(generate_response, next(api_key_cycle), api_config.model_type, api_config.api_url,
                                    prompt, **kwargs)): prompt
            for prompt in prompts
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_prompt),total = len(prompts)):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                yield prompt, result
            except Exception as exc:
                if '敏感' not in str(exc):
                    print(f"Prompt {prompt['index']} generated an exception: {exc}")
                    yield prompt, None


PromptType = TypeVar('PromptType')  # Define a type variable


def process_prompts_with_retries(api_config: ApiConfig, prompts: List[PromptType], max_retries=5, **kwargs) -> Iterator[
    Tuple[PromptType, Optional[str]]]:
    prompts_copy = prompts.copy()
    retries = defaultdict(int)
    while len(prompts_copy):
        input_prompts = prompts_copy.copy()
        prompts_copy.clear()
        for prompt, response in process_prompts(api_config, input_prompts, **kwargs):  # process failed prompts
            if response is None:
                retries[prompt['index']] += 1
                if retries[prompt['index']] < max_retries:
                    prompts_copy.append(prompt)
                    print(f"prompt {prompt['index']} tried = {retries[prompt['index']]}")
                else:
                    print(f"prompt {prompt['index']} reached max retries {max_retries}")
            else:
                yield prompt, response

def extract_json_content(text):
    # 使用正则表达式匹配 ```json\n 和 ``` 之间的内容
    pattern = r"```json\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def gte_embedding(texts: list[str]) -> np.ndarray:
    batch_size = 20
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        try:
            response = dashscope.TextEmbedding.call(
                api_key=GTE_API_KEY,
                model=dashscope.TextEmbedding.Models.text_embedding_v2,
                input=batch
            )
            embeddings.extend([dp['embedding'] for dp in response['output']['embeddings']])
        except Exception as e:
            print(e)
            time.sleep(5)
            response = dashscope.TextEmbedding.call(
                api_key=GTE_API_KEY,
                model=dashscope.TextEmbedding.Models.text_embedding_v2,
                input=batch
            )
            embeddings.extend([dp['embedding'] for dp in response['output']['embeddings']])

    return np.array(embeddings)

def llm_stream(prompt,stream=True):
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "user", "content":prompt},
        ],
        stream=stream
    )
    if stream:
        for chunk in response:
            print(chunk.choices[0].delta.content,end='')
    else:
        return response.choices[0].message.content.strip()

def llm_allinone(prompt,model = "deepseek-chat",stream=False):
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)    
    response = client.chat.completions.create(
        model=model,
        messages=[
        {"role": "user", "content":prompt},
        ],
        stream=stream
    )
    if stream:
        for chunk in response:
            print(chunk.choices[0].delta.content,end='')
    else:
        return response.choices[0].message.content.strip()
 
if __name__ == "__main__":
    print(llm_stream("hi",stream=False))
    exit()
    
    api_config = ApiConfig.from_json_file("./zhipu_config.json")
    system_prompt = ""
    # pompts formats
    # prompts = [{'index':0,'prompt':"What is the capital of France?",},
    #     {'index':1,'prompt':"What is the capital of France?",},
    #     {'index':2,'prompt':"Explain the theory of relativity.",},
    #     {'index':3,'prompt':"How does quantum computing work?",},
    #     {'index':4,'prompt':"Hi, say this is a test.",},
    # ]
    with open('./data/batch_data/output_file/to_generate_prompt_list.pkl','rb') as f:
        prompts = pickle.load(f)
    print(len(prompts))
    generated_prompts = []
    for prompt, response in process_prompts_with_retries(api_config, prompts, system_prompt=system_prompt):
        try:    
            res_dict = json.loads(extract_json_content(response)[0])
            prompt['response'] = res_dict
            generated_prompts.append(prompt)
        except:
            continue
    print(len(generated_prompts))
    with open('./data/batch_data/output_file/api_generated_prompts_v1.pkl','wb') as f:
        pickle.dump(generated_prompts,f)
