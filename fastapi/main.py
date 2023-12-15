import sys
from typing import Union
from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import contextlib

from loguru import logger

from user.router import userAPI 
from post.router import postAPI
from online.router import wsAPI
from aiwork.router import aiAPI, ml_models
from user.token import verify_token 
from config import get_env

# loguru setting
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")


# start with FastAPI
@logger.catch
@contextlib.asynccontextmanager
async def lifespan(app:FastAPI):
    for model in ml_models.values():
        if model is not None:
            model.load_model()
        else: 
            pass
         
    print("FastAPI Startup with ML-Models")
    
    yield

    ml_models.clear()
    print("FastAPI Shutdown")



app = FastAPI(lifespan=lifespan) # http & ws
# app = FastAPI(lifespan=lifespan, dependencies=[Depends(verify_token)])

# CROS Setting
app.add_middleware(CORSMiddleware, allow_origins=["*"], 
                   allow_credentials=True, 
                   allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(userAPI, prefix="/user", tags=["User Interfaces"], dependencies=[Depends(verify_token)])
app.include_router(postAPI, prefix="/post", tags=["POST Interfaces"], dependencies=[Depends(verify_token)])
app.include_router(aiAPI, prefix="/ai", tags=["AI Interfaces"], dependencies=[Depends(verify_token)])
# ws://localhost:port/ws/...
app.include_router(wsAPI, prefix="/ws", tags=["Websocket Interfaces"])



if __name__ == "__main__":
    env = get_env()
    # for debug
    uvicorn.run("main:app", host=env.Service_IP, port=env.Service_Port, reload=True)
    # for test
    # $ gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8088