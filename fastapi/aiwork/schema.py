from typing import Literal
from pydantic import BaseModel, Field

class TextInput(BaseModel):
    text: str = Field(..., max_length=512)

class TextInput2(TextInput):
    content: list[str] = Field(..., max_length=1000)

class TextInput3(BaseModel):
    texts: list[str] = Field(..., max_length=1000)

class TextPredict(BaseModel):
    text: str

class Text2Image(BaseModel):
    text: str
    ref_img: str | None = None 


class EmbeddingsResponse(BaseModel):
    model: Literal["gpt2xl"]
    data: list[float]


