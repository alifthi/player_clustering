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
        for j,path in enumerate(glob(MAIN_PICTURES_PATH+'/*')):
            name = path.split('/')[-1].split('.')[0]
            if int(name) in self.extracted:
                continue
            if j%100 == 0:
                print(j,'files readed')
            img = cv2.imread(path)
            blob = cv2.dnn.blobFromImage(img, swapRB=True)
            height, width, _ = img.shape
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            blank_mask = np.zeros((height, width), np.uint8)
            self.net.setInput(blob)
            boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
            detection_count = boxes.shape[2]
            height, width, _ = img.shape
            for i in range(detection_count):
                box = boxes[0, 0, i]
                class_id = int(box[1])
                score = box[2]
                if score < 0.8:
                    continue
                x = int(box[3] * width)
                y = int(box[4] * height)
                x2 = int(box[5] * width)
                y2 = int(box[6] * height)
                roi = blank_mask[y: y2, x: x2]
                roi_height, roi_width= roi.shape
                mask = masks[i, int(class_id)]
                mask = cv2.resize(mask, (roi_width, roi_height))
                _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = np.random.randint(0, 255, 3, dtype='uint8')
                color = [int(c) for c in color]
                for cnt in contours:
                    cv2.drawContours(roi, [cnt], 0, 1, -1)    
                for i , c_im in enumerate(roi):
                    for  j , v_im in enumerate(c_im):
                        img[i+y,j+x] = v_im and img[i+y,j+x]
                player = img[y:y2,x:x2]
                name = str(i) + '_' + path.split('/')[-1]
                cv2.imwrite(EXTRACTED_MASKED_PERSON_PATH + name,player)