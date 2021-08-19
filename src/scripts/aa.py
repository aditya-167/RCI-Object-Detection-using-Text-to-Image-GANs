import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


image = cv2.imread("yy.png")
    # red color boundaries [B, G, R]
lower = [np.mean(image[:,:,i] - np.std(image[:,:,i])/3 ) for i in range(3)]
upper = [250, 250, 250]

    # create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

ret,thresh = cv2.threshold(mask, 40, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


if len(contours) != 0:
    cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
    cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),5)

foreground = image[y:y+h,x:x+w]

cv2.imshow("image",image)
cv2.imshow("output",output)
cv2.imshow("foreground",foreground)
cv2.waitKey(0)
cv2.destroyAllWindows()