from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from aiwork.utils import env, ModelPathEnum


class NLLB200Translator:

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(ModelPathEnum.NLLB200)
        model = AutoModelForSeq2SeqLM.from_pretrained(ModelPathEnum.NLLB200)
        self.zh_en_translator = pipeline('translation', model=model, tokenizer=tokenizer,
                                         src_lang='zho_Hans', tgt_lang='eng_Latn', max_length=512)
        self.en_zh_translator = pipeline('translation', model=model, tokenizer=tokenizer,
                                         src_lang='eng_Latn', tgt_lang='zho_Hans', max_length=512)
        
    def en_to_zh(self, en_msg: str):
        return self.en_zh_translator(en_msg)
    
    def zh_to_en(self, zh_msg: str):
        return self.zh_en_translator(zh_msg)