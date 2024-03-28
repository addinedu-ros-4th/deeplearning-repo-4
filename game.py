import pyautogui
import numpy as np
import cv2
import time


from auto import *
from mqtt import *

def main():
    mqtt_client = MQTTClient()
    mqtt_client.connect()

    while True:
        img = Img()
        img.capture()
        cropped_images = img.get_chessboard_image()

        for i, cropped in enumerate(cropped_images):
            mqtt_client.publish_image(cropped)
            cv2.imshow(f'Cropped Image {i}', cropped)

        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        time.sleep(1)

if __name__ == "__main__":
    main()