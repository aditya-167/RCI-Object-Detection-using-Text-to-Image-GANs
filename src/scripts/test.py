#!/home/aditya/Capstone/ImaginationLearning/ros_python3_ws/ros_env_py3/bin/python

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge,CvBridgeError

from sensor_msgs.msg import Image


bridge = CvBridge()

def img_callback(ros_img):
    print("got an image")
    global bridge
    try:
        cv_img = bridge.imgmsg_to_cv2(ros_img,"bgr8")
    except CvBridgeError as e:
        print(e)
    
    # Load the aerial image and convert to HSV colourspace

    hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)

    # Define lower and uppper limits of what we call "brown"
    brown_lo=np.array([0,0,0])
    brown_hi=np.array([90,200,200])

    # Mask image to only select browns
    mask=cv2.inRange(hsv,brown_lo,brown_hi)

    # Change image to red where we found brown
    cv_img[mask>0]=(255,255,255)
    cv2.namedWindow("masked",cv2.WINDOW_NORMAL)
    cv2.imshow("masked",cv_img)
    cv2.waitKey(3)
    
    orig_img=cv_img.copy()
    hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    boxes=[]
    mask=cv2.inRange(hsv,brown_lo,brown_hi)
    cv_img[mask>0]=(255,255,255)
    brown_lo=np.array([82,82,82])
    brown_hi=np.array([192,192,192])
    mask=cv2.inRange(cv_img,brown_lo,brown_hi)
    cv_img[mask>0]=(255,255,255)
    original = cv_img.copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 100, 255, 1)
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # Iterate thorugh contours and filter for ROI
    image_number = 0
    maximum=0
    max_index=0;

    for i,c in enumerate(cnts):
        if(cv2.contourArea(c)<500):
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        new_image=ROI.copy()
        h=new_image[new_image<255]
      #cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        image_number += 1
    cv2.imshow("frame",orig_img)
    cv2.waitKey(3)


def main():
    rospy.init_node('test',anonymous=True)
    img_sub = rospy.Subscriber("/ILRobot/ILRobotcamera/image_raw",Image,img_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
        