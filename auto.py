import pyautogui
import numpy as np
import cv2


class Img:
    def __init__(self):
        self.image = None
        self.target_img = None

    def capture(self):
        pil_image = pyautogui.screenshot()
        opencv_image = np.array(pil_image)
        self.img = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        self.target_img = self.img.copy()

    def get_chessboard_image(self):
        if self.img is None:
            raise ValueError("Image not captured. Please run capture() method first.")
        
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
        cropped_images = []
        COLOR = (0, 200, 0) #Rectangle color

        for cnt in contours:
            if cv2.contourArea(cnt) > 100000:
                x, y, width, height = cv2.boundingRect(cnt)
                cv2.rectangle(self.target_img,(x, y),(x + width, y + height), COLOR, 2)
                cropped = self.target_img[y:y+height, x:x+width]
                cropped_images.append(cropped)

        return cropped_images





