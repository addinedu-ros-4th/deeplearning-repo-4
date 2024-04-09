import pyautogui
import numpy as np
import cv2


class Img:
    def __init__(self):
        self.image = None
        self.target_img = None

    def capture(self):
        pil_image = pyautogui.screenshot(region=(0,0, 1920, 1080))
        opencv_image = np.array(pil_image)
        self.img = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        self.target_img = self.img.copy()
        return self.img

    def automouse(self, move, top_left_corner, square_size):
        from_x = top_left_corner[0] + square_size * (ord(move[0]) - ord("a")) + square_size / 2
        from_y = top_left_corner[1] + square_size * (8 - int(move[1])) + square_size / 2
        to_x = top_left_corner[0] + square_size * (ord(move[2]) - ord("a")) + square_size / 2
        to_y = top_left_corner[1] + square_size * (8 - int(move[3])) + square_size / 2
        return from_x, from_y, to_x, to_y


def main():
    img = Img()
    img.capture()
    cropped_images = img.get_chessboard_image()

    for i, cropped in enumerate(cropped_images):
        cv2.imshow(f'Cropped Image {i}', cropped)
        print(type(cropped))

if __name__ == "__main__":
    main()




