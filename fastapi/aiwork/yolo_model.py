from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image
from loguru import logger

import pandas as pd
import numpy as np

from aiwork.utils import env, get_model_path, ModelPathEnum

class YOLO8Detection: # Local Model

    def load_model(self):
        self.model = YOLO(ModelPathEnum.YOLOv8)

    def attach_box_in_image(self, img: Image.Image)->Image:
         annotator = Annotator(np.array(img))
         predict = self.predict_single(img).get(0)
         predict.sort_values(by=['xmin'], ascending=True)

         for i, row in predict.iterrows():
             text = f"{row['name']}: {int(row['confidence']*100)}%"
             bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
             annotator.box_label(bbox, text, color=colors(row['class'], True))
    
         return Image.fromarray(annotator.result())



    def predict_single(self, img:Image.Image):
        if not self.model:
            raise RuntimeError("model is not loaded")
        
        results = self.predict(img=img)

        predict_list = {}
        for index,result in enumerate(results):
            predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
            predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
            predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
            predict_bbox['name'] = predict_bbox["class"].replace(results[0].names)

            predict_list[index] = predict_bbox
        
        return predict_list
        

    def predict(self, img: Image.Image | list) -> Results:
        return self.model.predict(source=img, conf=env.YOLO_DETECTION_MODEL_CONF,
                                    flipud=env.YOLO_DETECTION_MODEL_FLIPUD, 
                                    fliplr=env.YOLO_DETECTION_MODEL_FLIPLR,
                                    mosaic=env.YOLO_DETECTION_MODEL_MOSAIC)  


        
