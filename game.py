import pyautogui
import numpy as np
import cv2
from auto import Image




def main():
    img = Image()
    img.capture()
    cropped_images = img.get_chessboard_image()

    for i, cropped in enumerate(cropped_images):
        cv2.imshow(f'Cropped Image {i}', cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# while True: