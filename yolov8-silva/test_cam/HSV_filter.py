import cv2
import numpy as np


def add_HSV_filter(frame, camera):
    blur = cv2.GaussianBlur(frame,(5,5),0) # Blurring the frame
    # Converting RGB to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    l_b_r = np.array([0, 184, 36])        # Lower limit for red ball
    u_b_r = np.array([255, 255, 255])       # Upper limit for red ball
    l_b_l = np.array([0, 184, 36])        # Lower limit for red ball
    u_b_l = np.array([255, 255, 255])       # Upper limit for red ball

    if(camera == 1):
        mask = cv2.inRange(hsv, l_b_r, u_b_r)
    else:
        mask = cv2.inRange(hsv, l_b_l, u_b_l)


    # Morphological Operation - Opening - Erode followed by Dilate - Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask
