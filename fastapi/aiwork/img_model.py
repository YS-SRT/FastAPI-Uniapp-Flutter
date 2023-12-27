import torch
from torch import autocast
from safetensors import safe_open
from diffusers import StableDiffusionXLImg2ImgPipeline, DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image
from PIL import Image

from aiwork.utils import env, ModelPathEnum

class Text2ImgBaseGenerator: #stable-diffusion-xl-1.0

    def load_model(self):
        pipe = StableDiffusionXLPipeline.from_single_file(ModelPathEnum.SDXL,
                                                        # torch_dtype=torch.float16, 
                                                        use_safetensors=True, 
                                                        variant="fp16",
                                                        original_config_file=ModelPathEnum.SDXL_CONF
                                                        )
        
        # pipe = pipe.to("cuda")
        pipe = pipe.to("cpu")
        pipe.enable_model_cpu_offload()
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe = pipe
    
    def predict(self, prompt:str) -> Image:
        return self.pipe(prompt).images[0]
    
class Text2ImgRefinerGenerator: #stable-diffusion-xl-1.0

    def load_model(self):
        pipe = StableDiffusionXLPipeline.from_single_file( ModelPathEnum.SDXL_REFINER, 
                                                           # torch_dtype=torch.float16, 
                                                           variant="fp16", 
                                                           use_safetensors=True,
                                                           original_config_file=ModelPathEnum.SDXL_REFINER_CONF)
        # pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe = pipe

    def predict_with_net(self, prompt:str, url:str) -> Image:
        ref_img = load_image(url)
        return self.predict(prompt, ref_img)

    def predict_with_img(self, prompt:str, ref_img:Image) -> Image:
        return self.pipe(prompt, image= ref_img.convert("RGB")).images
        
        
class Text2ImgMixedGenerator(Text2ImgBaseGenerator): #stable-diffusion-xl-1.0

    def load_model(self):
        base = super().load_model()

        refiner = StableDiffusionXLPipeline.from_single_file(ModelPathEnum.SDXL_REFINER,
                                                             text_encoder_2=base.text_encoder_2,
                                                             vae=base.vae,
                                                             # torch_dtype=torch.float16,
                                                             use_safetensors=True,
                                                             variant="fp16",
                                                             original_config_file=ModelPathEnum.SDXL_REFINER_CONF)
        # refiner.to("cuda")
        refiner.enable_model_cpu_offload()
        refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        
        self.base = base
        self.refiner = refiner
    
    def predict(self, prompt:str, n_steps=40, high_noise_frac=0.8) -> Image:
        image = self.base(prompt=prompt, 
                          num_inference_steps=n_steps,
                          denoising_end=high_noise_frac,
                          output_type="latent").images
        image = self.refiner(prompt=prompt, 
                             num_inference_steps=n_steps, 
                             denoising_start=high_noise_frac,
                             image=image).images[0]
        return image


if __name__ == "__main__":
   base_generator = Text2ImgBaseGenerator()
   base_generator.load_model()
   base_generator.predict("a cat sit in boat with a little girl")
      



