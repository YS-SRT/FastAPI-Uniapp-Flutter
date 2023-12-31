from fastapi import FastAPI, APIRouter, UploadFile, File
from contextlib import asynccontextmanager

import json
from PIL import Image, ImageDraw2, ImageFont
from fastapi.responses import StreamingResponse

from aiwork.yolo_model import YOLO8Detection
from aiwork.nllb_model import NLLB200Translator
from aiwork.img_model import Text2ImgBaseGenerator, Text2ImgRefinerGenerator, Text2ImgMixedGenerator
from aiwork.gpt2_model import GPT2XLGenerator
from aiwork.schema import Text2Image, TextInput, TextInput2, TextInput3, TextPredict, EmbeddingsResponse
from aiwork.utils import env, img_to_bytes

ml_models = { 
              "yolo_detection": YOLO8Detection() if env.ENABLE_YOLO else None,
              "nllb_translator": NLLB200Translator() if env.ENABLE_NLLB else None,
              "sdxl_refiner":Text2ImgRefinerGenerator() if env.ENABLE_SDXL_REFINER else None,
              "sdxl_mixed":Text2ImgMixedGenerator() if env.ENABLE_SDXL_MIXED else None,
              "sdxl_base": Text2ImgBaseGenerator() if env.ENABLE_SDXL_BASE else None,
              "gpt2xl": GPT2XLGenerator() if env.ENABLE_GPT2XL else None
             }

aiAPI = APIRouter()

@aiAPI.post("/en2zh", response_model=TextPredict)
async def en2zh_translator(input:TextInput):
    result = ml_models["nllb_translator"].en_to_zh(input.text)
    return TextPredict(text= result[0].get("translation_text"))

@aiAPI.post("/zh2en", response_model=TextPredict)
async def en2zh_translator(input:TextInput):
    result = ml_models["nllb_translator"].zh_to_en(input.text)
    return TextPredict(text= result[0].get("translation_text"))

@aiAPI.post("/yolo_label")
async def detect_image_objects(img: UploadFile = File(...)) :
    image = Image.open(img.file)
    predict = ml_models["yolo_detection"].predict_single(image)
    return {"detect_result": json.loads(predict.get(0).to_json(orient="records"))}

@aiAPI.post("/yolo_label_show")
async def detect_image_objects(img: UploadFile = File(...)) :
    image = Image.open(img.file)
    predict_img = ml_models["yolo_detection"].attach_box_in_image(image)
    return StreamingResponse(content=img_to_bytes(predict_img), media_type="image/jpeg")

@aiAPI.post("/sdxl_base_text2img")
async def sdxl_base_text2img(input:TextInput):
    predict_img = ml_models["sdxl_base"].predict(input.text)
    return StreamingResponse(content=img_to_bytes(predict_img), media_type="image/jpeg")
    
@aiAPI.post("/sdxl_refiner_img2img")
async def sdxl_Refiner_img2img(input:TextInput, img: UploadFile = File()):   
    predict_img = ml_models["sdxl_refiner"].predict_with_img(input.text, Image.open(img.file))                  
    return StreamingResponse(content=img_to_bytes(predict_img), media_type="image/jpeg")

@aiAPI.post("/sdxl_mixed_text2img")
async def sdxl_base_text2img(input:TextInput):
    predict_img = ml_models["sdxl_mixed"].predict(input.text)
    return StreamingResponse(content=img_to_bytes(predict_img), media_type="image/jpeg")

@aiAPI.post("/gpt2xl_gc")
async def gpt2xl_gc(input:TextInput):
    return {"gpt2xl": ml_models["gpt2xl"].predict(input.text)}

@aiAPI.post("/gpt2xl_embedding_gc")
async def gpt2xl_embedding_gc(input:TextInput2):
    return {"gpt2xl": ml_models["gpt2xl"].predict_qa(input.text, input.content)}

@aiAPI.post("/gpt2xl_embedding")
async def gpt2xl_embedding(input:TextInput):
    embeddings = ml_models["gpt2xl"].get_input_embedding(input.text)
    return EmbeddingsResponse(model="gpt2xl",data=embeddings)

    


