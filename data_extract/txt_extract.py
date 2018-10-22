from constant import DATAPATH

import pytesseract
import cv2
import os


LANGUAGE = 'eng' # 'chi_sim'

def text_extract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang=LANGUAGE)
    return text


def main():
    INPUT_IMG = 'table2.png'
    input_img_path = os.path.join(DATAPATH, INPUT_IMG)
    image = cv2.imread(input_img_path)
    print(text_extract(image))


if __name__ == '__main__':
    main()
