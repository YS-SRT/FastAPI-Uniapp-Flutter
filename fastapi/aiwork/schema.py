from pydantic import BaseModel, Field

class TextInput(BaseModel):
    text: str = Field(..., max_length=512)

class TextPredict(BaseModel):
    text: str

class Text2Image(BaseModel):
    text: str
    ref_img: str | None = None 
    


