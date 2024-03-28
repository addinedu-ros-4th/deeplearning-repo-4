from ultralytics import YOLO
from mqtt import *
import time
import cv2
import pandas as pd
import torch
import numpy as np

def main():

    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")  # or yolov5n - yolov5x6, custom
    mqtt_client = MQTTClient()
    client = mqtt_client.get_client()
    client.on_message = mqtt_client.on_message
    mqtt_client.connect()
    
    while True:
        mqtt_client.image_ready.wait()
        img = mqtt_client.get_last_image()
        if img is not None:
            #results = model(img)  # inference
            #crops = results.crop(save=True)  # cropped detections dictionary
            results = model(img)  # inference
            print(results.pandas().xyxy[0])
        mqtt_client.image_ready.clear()

        
if __name__ == "__main__":
    main()