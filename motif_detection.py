import os
import ftfy
from PIL import Image
import traceback
import sys
from ultralytics import YOLO

#WIP: Detect motifs in enhanced greyscale TVT visualisations


def log_exception(exception: BaseException, expected: bool = True):
        output = "[{}] {}: {}".format(
            'EXPECTED' if expected else 'UNEXPECTED', type(exception).__name__, exception)


        exc_type, exc_value, exc_traceback = sys.exc_info()

        out_text=f"{output}\n{traceback.format_exception(exc_type, exc_value, exc_traceback)}"
        return out_text
    
def check_dirs(out_dir,class_list):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        for subdir in class_list:
            os.makedirs(f"{out_dir}/{subdir}")
        
    
        
def extract_images(file,data_dir,out_dir,min_conf,min_iou,size,model):
    if os.path.splitext(file)[1] in ['.png','.jpg','.jpeg','.tif','.tiff']:
        img=Image.open(ftfy.fix(f"{data_dir}/{file}"))
        results=model(img,conf=min_conf,iou=min_iou,imgsz=size,save_txt=True,save_conf=True,save=True,filename=f"{out_dir}/{file}_anno.png")
        for result in results:
            class_map=result.names
            for i,box in enumerate(zip(result.boxes.cls,result.boxes.xyxy)):
                out_img=img.crop(box[1].tolist())
                cls=class_map[box[0].tolist()]
                out_img.save(f"{out_dir}/{cls}/{os.path.splitext(file)[0]}_{i}.png")

for model_type in []:
    model=YOLO(f"{model_type}")
    class_list = model.names