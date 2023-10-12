import cv2 
import numpy as np
from glob import glob
from config import EXTRACTED_MASKED_PERSON_PATH, COCO_NAMES, COCO_PB,\
                    COCO_PBTXT, MAIN_PICTURES_PATH, MASKED, EXTRACTED_PERSON_PATH
class extract_persons:
    def __init__(self) -> None:
        if MASKED:
            self.pathToSave=EXTRACTED_MASKED_PERSON_PATH
        else:
            self.pathToSave=EXTRACTED_PERSON_PATH
        self.extracted=None   
    def check_extracted(self):
        for path in glob(self.pathToSave+'*'):
            name = path.split('/')[-1].split('.')[0].split('_')[-1]
            self.extracted.append(int(name))  
    def load_extractor(self):
        self.net = cv2.dnn.readNetFromTensorflow(COCO_PB,COCO_PBTXT)
    def extract_persons(self):
        pass