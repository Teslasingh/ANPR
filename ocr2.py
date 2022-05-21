
import numpy as np
import cv2
import pytesseract
import time
import imutils
from PIL import Image
cap = cv2.VideoCapture(2)
fnt = cv2.FONT_HERSHEY_DUPLEX
text = '_'
fount= '_'
while True:
    #ret, image =cap.read()
    #key = cv2.waitKey(1) & 0xFF

    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
        break
    if screenCnt is None:
        detected = 0
        print("No contour detected")

    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        count =0
        print("Detected Number is:", text)
        for i in range(0, len(text)):
            if(text[i] != ' '):
                count = count + 1
        print("Total number of characters in a string: " + str(count));  
        
        if(count>6 and count < 25 ):
            image = cv2.putText(image, text, (300, 300), fnt, 1, (5, 200, 150), 1)
            fount = text
        cv2.imshow('Cropped', Cropped)
    image = cv2.putText(image,fount , (300, 300), fnt, 1, (5, 200, 150), 1)
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
