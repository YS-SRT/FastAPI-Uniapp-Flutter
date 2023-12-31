import os, sys
from enum import Enum


cur_path = os.getcwd()

def add_upper_dir_in_path():
    upper_dir = os.path.dirname(cur_path)
    if upper_dir not in sys.path:
         sys.path.append(upper_dir)

class DataPathEnum(str, Enum):
    ZH_HANDWRITTING_IMG_DIR = "chinese_handwritting/images"
    ZH_HANDWRITTING_CSV = "chinese_handwritting/chinese_mnist.csv"
    ZH_HANDWRITTING_TFRECORD = "chinese_handwritting/chinese_mnist.tfrecords"
    LANGCHAIN_DATA_DIR = "knowledge"
    LANGCHAIN_DATA_VECTOR_DIR = "knowledge/vector_store"
    

    def __str__(self):
        return os.path.join(cur_path, "data", self.value)
    
