from ultralytics import YOLO
import time
import cv2
import pandas as pd

import numpy as np
from PIL import Image
import pyautogui
from auto import *

def main():

    
    model = YOLO("best_v8.pt")
    img = "/home/addinedu/Downloads/test.png"
    result = model.predict(img)
    boxes = result[0].boxes # Boxes object for bounding box outputs
    xywh = boxes.xywh
    classes = boxes.cls
    conf = boxes.conf

    xywh_np = xywh.cpu().numpy()
    classes_np = classes.cpu().numpy()
    conf_np = conf.cpu().numpy()
    classes_df = pd.DataFrame(classes_np, columns=["class"])
    conf_df = pd.DataFrame(conf_np, columns=["confidence"])
    xywh_df = pd.DataFrame(xywh_np, columns=["x_mid", "y_mid", "width", "height"])
    df = pd.concat([classes_df, conf_df, xywh_df], axis=1)
    print(df)

    image = cv2.imread(img)

    for index, row in df.iterrows():
        x_mid = int(row['x_mid'])
        y_mid = int(row['y_mid'])
        width = int(row['width'])
        height = int(row['height'])
        
        # 바운딩 박스의 우측 하단 모서리 계산
        x_min = round(x_mid - width / 2)
        x_max = round(x_mid + width / 2)
        y_min = round(y_mid - height / 2)
        y_max = round(y_mid + height / 2)
        
        # 이미지에 바운딩 박스 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text = f"Class: {int(row['class'])}, Conf: {row['confidence']:.2f}"
    
        # 이미지에 클래스와 신뢰도 텍스트 추가
        cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    cropped_images = []
    COLOR = (0, 200, 0) #Rectangle color

    frame = result[0].plot(font_size = 10, pil = True)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100000:
            x, y, width, height = cv2.boundingRect(cnt)
            # cv2.rectangle(image,(x, y),(x + width, y + height), COLOR, 2)
            cropped = frame[y:y+height, x:x+width]


    cv2.imshow('Image with Bounding Boxes', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()