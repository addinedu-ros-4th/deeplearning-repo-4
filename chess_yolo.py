import torch
from mqtt import *
import time

def main():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")  # or yolov5n - yolov5x6, custom
    mqtt_client = MQTTClient()
    client = mqtt_client.get_client()
    client.on_message = mqtt_client.on_message
    mqtt_client.connect()
    
    while True:
        img = mqtt_client.get_last_image()
        if img is not None:
            results = model(img)

            #results = model(img)  # inference
            #crops = results.crop(save=True)  # cropped detections dictionary
            results = model(img)  # inference
            results.pandas().xyxy[0]  # Pandas DataFrame
            print(results.pandas().xyxy[0])
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()