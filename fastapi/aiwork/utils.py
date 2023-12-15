import os, io
from enum import Enum
from PIL import Image
from config import get_env


env = get_env()
cur_dir = os.getcwd()

class ModelPathEnum(str, Enum):
    YOLOv8 = "ml_models/yolov8n.pt"
    NLLB200 = "ml_models/NLLB-200-600M"
    SDXL = "ml_models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0_0.9vae.safetensors"
    SDXL_CONF="ml_models/stable-diffusion-xl-base-1.0/sd_xl_base.yaml"
    SDXL_REFINER = "ml_models/stable-diffusion-xl-base-1.0/sd_xl_refiner_1.0_0.9vae.safetensors"
    SDXL_REFINER_CONF="ml_models/stable-diffusion-xl-base-1.0/sd_xl_refiner.yaml"

    def __str__(self):
        return os.path.join(os.getcwd(), self.value)

def img_to_bytes(img:Image):
    resp_img = io.BytesIO()
    img.save(resp_img, format='JPEG', quality=env.DETECTION_MODEL_RESP_IMG_QUARITY) 
    resp_img.seek(0) 
    return resp_img


