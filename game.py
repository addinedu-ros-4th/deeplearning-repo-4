import pyautogui
import numpy as np
import cv2
import time

from auto import *


def main():
    while True:
        img = Image()
        img.capture()
        cropped_images = img.get_chessboard_image()

        for i, cropped in enumerate(cropped_images):
            cv2.imshow(f'Cropped Image {i}', cropped)

        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        time.sleep(1)

if __name__ == "__main__":
    main()