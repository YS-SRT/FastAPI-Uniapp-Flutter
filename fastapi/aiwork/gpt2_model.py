import torch
from torch import autocast
from safetensors import safe_open
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

from aiwork.utils import ModelPathEnum, get_qa_template

class GPT2XLGenerator():
    
    def load_model(self):
        tokenizer = GPT2Tokenizer.from_pretrained(ModelPathEnum.GPT2XL, local_files_only=True)
        tokenizer.padding_side="left"
        model = GPT2LMHeadModel.from_pretrained(ModelPathEnum.GPT2XL, local_files_only=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        


    def predict(self, input_text:str):   
        return self._predict(input_text)


    def predict_qa(self, input_text:str, input_content:list[str]):  
        contents = "\n".join([item for item in input_content])
        template = get_qa_template().replace("{question}", input_text).replace("{context}", contents)
        return self._predict(template)
        
    
    def _predict(self, input):
        count = len(input) 
        input_ids = self.tokenizer.encode(input, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=200, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text[count:-1].strip()
    
    def get_input_embedding(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**input_ids)
        embedding = output[0][0].mean(dim=0).tolist()
        return embedding