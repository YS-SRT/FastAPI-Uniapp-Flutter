from typing import Any
from random import randint
import matplotlib.pyplot as plt, os, pandas as pd
from PIL import Image

# output chinese word
import matplotlib
matplotlib.rc("font",family='SimHei')

from aitrain.utils import DataPathEnum

IMAGE_DIR = str(DataPathEnum.ZH_HANDWRITTING_IMG_DIR)

# image = 64 * 64
class HWData():

    def __init__(self) -> None:
        self.image_files = os.listdir(IMAGE_DIR)
        self.character_str ="零一二三四五六七八九十百千万亿"
        self.image_folder:str = IMAGE_DIR

    def get_image_path_vs_lable(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_file)
        label = int(image_file.split(".")[0].split("_")[-1]) -1
        return label, image_path   
    
    def plot_image(self, index):
        label, image_path = self.get_image_path_vs_lable(index)
        image = Image.open(image_path)

        plt.title("label: " + str(label) + "/" + self.character_str[label])
        plt.imshow(image)