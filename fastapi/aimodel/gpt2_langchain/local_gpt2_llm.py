from typing import Any, List, Optional
from langchain_core.outputs import LLMResult
from langchain_core.embeddings import Embeddings
import requests
from enum import Enum
from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import BaseModel

class APIPathEnum(str, Enum):

    simple_url = "http://127.0.0.1:8088/ai/gpt2xl_gc"
    content_url = "http://127.0.0.1:8088/ai/gpt2xl_embedding_gc"
    embedding_url = "http://127.0.0.1:8088/ai/gpt2xl_embedding"


class LocalGPT2XL(LLM):
    
    def __init__(self):
        super().__init__()

    
    def _llm_type(self) -> str:
        return "gpt2xl_model"

    
    def _invoke_api(self, url:str, prompt:str, content:list[str]|None = None):

        req_dict = {"text": prompt}
        if content is not None:
            req_dict["content"] = content
        
        headers = {"Context_Type": "application/json"}
        resp = requests.post(url, json=req_dict, headers= headers)

        if resp.status_code == 200:
            return resp.json()["gpt2xl"]
        else:
            return None
        

    def _call(self, prompt: str, stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> str:

        if kwargs is None or kwargs["content"] is None:
            resp = self._invoke_api(APIPathEnum.simple_url.value, prompt)
        else:
            resp = self._invoke_api(APIPathEnum.content_url.value, prompt, content = kwargs["content"])
        
        return  resp if resp is not None else "gpt2xl_error"
    

class LocalGPT2Embedding(Embeddings):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    

    def _invoke_api_once(self, txt:str):
        req_dict = {"text": txt}
        headers = {"Context_Type": "application/json"}
        resp = requests.post(APIPathEnum.embedding_url.value, json=req_dict, headers= headers)
        if resp.status_code == 200:
             return resp.json()["data"]
        else:
             return None

    def embed_query(self, text: str) -> List[float]:
        resp = self._invoke_api_once(text)
        return resp
    
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embedding_list = []
        for text in texts:
            resp = self._invoke_api_once(text)
            embedding_list.append(resp)
        return embedding_list



        

    
        
        
      
        
        
    

   
