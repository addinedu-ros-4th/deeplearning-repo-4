from ultralytics import YOLO
from mqtt import *
import time
import cv2
import pandas as pd
import torch
import numpy as np

def main():
    model = YOLO("best_v8.pt")

    img = "/home/addinedu/Downloads/test.png"
    results = model(img)
    print(results)  # results 객체가 제공하는 속성과 메서드를 확인합니다.

        
if __name__ == "__main__":
    main()