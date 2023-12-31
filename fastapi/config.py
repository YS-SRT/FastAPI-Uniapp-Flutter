from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
     # Service
     Service_Port:int
     Service_IP:str

     # Token Setting
     TOKEN_SECRET_KEY:str
     TOKEN_ALGORITHM:str
     TOKEN_ACCESS_TOKEN_EXPIRE_MINUTES:int

     # Validate Pattern
     REGEX_PHONE:str
     REGEX_EMAIL:str

     # Redis Server
     REDIS_CONNECT_STRING:str

     # Mongodb Server
     MGDB_CONNECT_STRING:str

     # Postgresql Server
     PQDB_CONNECT_STRING:str

     # Minio Service
     MINIO_SERVICE_DOMAIN:str
     MINIO_ACCESS_KEY:str
     MINIO_SECRET_KEY:str

     # celery setting
     CELERY_BACKEND_REDIS:str
     CELERY_BROKER_REDIS:str

     # dramatiq setting
     DRAMATIQ_BROKER_REDIS:str

     
     # AI-Models
     ENABLE_YOLO:bool
     ENABLE_NLLB:bool
     ENABLE_SDXL_BASE:bool
     ENABLE_SDXL_REFINER:bool
     ENABLE_SDXL_MIXED:bool
     ENABLE_GPT2XL:bool


     DETECTION_MODEL_RESP_IMG_QUARITY:int

     YOLO_DETECTION_MODEL_CONF:float
     YOLO_DETECTION_MODEL_FLIPUD:float
     YOLO_DETECTION_MODEL_FLIPLR:float
     YOLO_DETECTION_MODEL_MOSAIC:float

     # gpt2xl
     QA_PROMPT_TEMPLATE:str = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
     {context}
     Question:{question}"""

     model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_env():
    return Settings()
