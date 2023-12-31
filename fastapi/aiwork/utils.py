import os, io
from enum import Enum
from PIL import Image
from config import get_env

env = get_env()

class ModelPathEnum(str, Enum):
    YOLOv8 = "yolov8n.pt"
    NLLB200 = "NLLB-200-600M"
    Yi34BChat = "Yi-34b-Chat"
    SDXL = "stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors"
    SDXL_CONF="stable-diffusion-xl-base-1.0/sd_xl_base.yaml"
    SDXL_REFINER = "stable-diffusion-xl-base-1.0/sd_xl_refiner_1.0_0.9vae.safetensors"
    SDXL_REFINER_CONF="stable-diffusion-xl-base-1.0/sd_xl_refiner.yaml"
    GPT2XL = "gpt2-xl"

    def __str__(self):
        return os.path.join(os.getcwd(), "ml_models", self.value)

def img_to_bytes(img:Image):
    resp_img = io.BytesIO()
    img.save(resp_img, format='JPEG', quality=env.DETECTION_MODEL_RESP_IMG_QUARITY) 
    resp_img.seek(0) 
    return resp_img


def get_qa_template():
    return env.QA_PROMPT_TEMPLATE