from ultralytics import YOLO
import time
import cv2
import pandas as pd

import numpy as np
from PIL import Image
import pyautogui
from auto import *

def main():
    time.sleep(2)
    
    model = YOLO("best_v8.pt")
    img = Img()
    frame = img.capture()
    # cropped_images = img.get_chessboard_image()

    # for i, cropped in enumerate(cropped_images):
    #     frame = cropped
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     print(type(frame))
    results = model.predict(frame)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image

if __name__ == "__main__":
    main()