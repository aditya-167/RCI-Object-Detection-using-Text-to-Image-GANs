#!/home/aditya/Capstone/ImaginationLearning/ros_python3_ws/ros_env_py3/bin/python
import rospy
import cv2
from std_msgs.msg import String 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

import sys

bridge = CvBridge()

def img_callback(ros_img):

    print("got an image")
    global bridge

    try:
        cv_img = bridge.imgmsg_to_cv2(ros_img,"bgr8")
    except CvBridgeError as e:
        print(e)
    
    (rows,cols,channels) = cv_img.shape
    if(cols>200 and rows > 200):
        cv2.circle(cv_img,(100,100),90,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cv_img,'ADITYA',(10,350),font,1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("ROS IMAGE",cv_img)
    cv2.waitKey(3)

def main():
    rospy.init_node('VideoTest',anonymous=True)
    img_sub = rospy.Subscriber("/ILRobot/ILRobotcamera/image_raw",Image,img_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
        