import os, sys
from enum import Enum

cur_path = os.getcwd()

def add_upper_dir_in_path():
    upper_dir = os.path.dirname(cur_path)
    if upper_dir not in sys.path:
         sys.path.append(upper_dir)

class DataPathEnum(str, Enum):
    ZH_HANDWRITTING_IMG_DIR = "chinese_handwritting/images"
    GPT2XL_TRAIN_DATA_DIR = "gpt2xl"
    MODEL_CHECKPOINT_DIR = "checkpoints"

    def __str__(self):
        return os.path.join(cur_path, "data", self.value)
    

class ModelPathEnum(str, Enum):
    GPT2XL = "gpt2-xl"

    def __str__(self):
        return os.path.join(os.path.dirname(cur_path), "ml_models", self.value)