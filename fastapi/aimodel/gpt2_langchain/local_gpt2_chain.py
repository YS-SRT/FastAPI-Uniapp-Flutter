import os
from typing import List, Dict, Any, Optional
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.outputs import LLMResult
from langchain_core.prompts.string import PromptValue
from functools import lru_cache

from aimodel.utils import DataPathEnum
from aimodel.gpt2_langchain.local_gpt2_llm import *

VDB_SIGN = "myknowledge"
VDB_DIR = str(DataPathEnum.LANGCHAIN_DATA_VECTOR_DIR)
QA_TEMPLATE="""QA Processing: >>>
{qa_history}
Q:{question}
A:"""

class LocalVectorStore():
   
   @classmethod
   @lru_cache
   def init_vector_db(cls, embedding: Embeddings) -> None:
       if os.path.exists(os.path.join(VDB_DIR, VDB_SIGN + ".faiss")):
           return FAISS.load_local(VDB_DIR, embedding, VDB_SIGN, distance_strategy=DistanceStrategy.COSINE)
           # return Chroma(VDB_SIGN,embedding,VDB_DIR)
       else:
           return cls.load_docs_into_db(embedding)
       
   
   @classmethod
   def load_docs_into_db(self, embedding)->FAISS:
       files = map(lambda f: os.path.join(dir, f), os.listdir(VDB_DIR))
       files = filter(lambda f: os.path.isfile(f) and f.endswith(".txt"), files)
       
       docs = []
       for f in files:
           text_splitter = CharacterTextSplitter()
           docs += TextLoader(f, "utf8").load_and_split(text_splitter)

       vector_db = FAISS.from_documents(docs, embedding)
       vector_db.save_local(VDB_DIR, VDB_SIGN)

       # vector_db = Chroma.from_documents(docs, embedding)
       # vector_db.persist()

       return vector_db
       

class LocalKnowledgeQAChain(LLMChain):
    
    local_vdb = LocalVectorStore.init_vector_db(LocalGPT2Embedding())

    def __init__(self) -> None:
        qa_prompt = PromptTemplate(input_variables=["question"], 
                                   template=QA_TEMPLATE)
        super().__init__(llm=LocalGPT2XL(),
                         memory=ConversationBufferMemory(memory_key="qa_history"),
                         prompt=qa_prompt,
                         output_key="gpt2xl",
                         return_final_only=True,
                         verbose=True)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        content=None
        # search in vector_db
        related_contents = self.local_vdb._similarity_search_with_relevance_scores(inputs["question"])
        if related_contents is not None:
            content = [doc.page_content for doc, _ in related_contents]

        # search from other sources
        other_contents = self._search_from_sources(inputs)
        if other_contents is not None:
            content = content + other_contents if content is not None else other_contents
        
        # invoke LLM
        resp = self.llm.generate(prompts=[self._build_template(inputs)], content=content)
        return self.create_outputs(resp)[0]

    def _build_template(self, inputs: Dict[str, Any]):
        prompts, _ = self.prep_prompts([inputs])
        return prompts[0].to_string()

    def _search_from_sources(self, inputs: Dict[str, Any]) ->list[str]:
        pass # from DB or ES

    # access local LLM directly
    def ask(self, question:str):
        # search in vector_db
        related_contents = self.local_vdb.similarity_search_with_score(question)
        if related_contents is not None:
            content = [doc.page_content for doc, _ in related_contents]
        # invoke LLM API
        resp = self.llm._call(prompt=question, content=content)
        return resp
    

