#!/home/aditya/Capstone/ImaginationLearning/ros_python3_ws/ros_env_py3/bin/python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from math import radians, degrees 
from actionlib_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import os
from ILRobot.srv import DetectObject
from ILRobot.srv import DetectObjectRequest
from ILRobot.srv import DetectObjectResponse
import sys
from geometry_msgs.msg import Point
import speech_recognition as asr
import numpy as np
import cv2 as cv
image_number = 0


bridge = CvBridge()

OBJECTS = ['bananas','yellow fruit','yellow colored fruit',
    'banana','banana fruit','yellow fresh banana','spoilt banana with black spots',
    'oranges','orange colored fruit','orange fruit',
    'orange with green leaf','fresh oranges with green leaf','fresh orange with green leaf',
    'apple','red round apple','red round apple with leaf',
    'round apple with green leaf','yellowish apple','fresh yellowish apple',
    'fresh round apple with green leaf','green apple','green apple with leaf',
    'spoilt apple with black spots','red round yellowish apple',
    'apples','orange']

LOCATIONS = ['drawingroom','hall','livingroom','kitchen','bedroom','bed room','living room','drawing room']

COORDINATES = {"drawingroom":(-6.03833436966,0.830349624157),"drawing room":(-6.03833436966,0.830349624157),"hall":(-6.03833436966,0.830349624157),
            "kitchen":(-1.86791014671, -5.44469451904),"livingroom":(4.08805704117,2.35057401657),"living room":(4.08805704117,2.35057401657),
            "bedroom":(3.63946032524,-1.65342497826),"bed room":(3.63946032524,-1.65342497826)}


def obj_detect_client(name):
    rospy.wait_for_service('Detect_Object_Search_Service')
    try:
        detect_obj = rospy.ServiceProxy('Detect_Object_Search_Service',DetectObject)
        resp = detect_obj(name)
        return resp.success
    except:
        pass

def stop():
    vel_twist = Twist()
    vel_twist.linear.x = 0
    vel_twist.linear.y = 0
    vel_twist.linear.z = 0

    vel_twist.angular.x = 0
    vel_twist.angular.y = 0
    vel_twist.angular.z = 0

    velocity_pub.publish(vel_twist)
    print("Robot has stopped!")

def img_callback(ros_img):
    print("got frame!")
    global bridge
    global image_number
    try:
        cv_img = bridge.imgmsg_to_cv2(ros_img,"bgr8")
    except CvBridgeError as e:
        print(e)
    orig_img=cv_img.copy()
    height,width,_ = orig_img.shape
    hsv=cv.cvtColor(cv_img,cv.COLOR_BGR2RGB)
    # Define lower and uppper limits of what we call "brown"
    brown_lo=np.array([0,0,0])
    brown_hi=np.array([105,220,220])
    #brown_hi=np.array([60,150,220])

    # Mask image to only select browns
    mask=cv.inRange(hsv,brown_lo,brown_hi)

    # Change image to red where we found brown
    cv_img[mask>0]=(255,255,255)
    brown_lo=np.array([72,72,72])
    brown_hi=np.array([255,200,200])

    # Mask image to only select browns
    mask=cv.inRange(cv_img,brown_lo,brown_hi)
    cv_img[mask>0]=(255,255,255)
    #cv.namedWindow("frame",cv.WINDOW_NORMAL)

    original = cv_img.copy()
    gray = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blurred, 120, 255, 20)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv.dilate(canny, kernel, iterations=1)

    # Find contours
    cnts = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and filter for ROI
    max=0
    max_index=0

    for i,c in enumerate(cnts):
        if(cv.contourArea(c)<350):
            continue
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(orig_img, (x, y), (x + w, y + h), (36,255,12), 2)

        ROI = original[y:y+h, x:x+w]
        new_image=ROI.copy()

        #if model(new_image)==banan:
        #    pass
        #h=new_image[new_image<255]
        cv.imwrite("/home/aditya/Capstone/ImaginationLearning/src/ILRobot/src/Img/ROI_{}.png".format(image_number), ROI)
        image_number += 1
    
    cv.imshow("frame",orig_img)
    cv.waitKey(5)

def move_and_detect(x,y):
    if moveGoal(x,y):
        stop()
        rate.sleep()
        if obj_detect_client("detect object"):
            print("returning from service\n")
            stop()
            return True
    
    else:
        stop()

        rate.sleep()
        if obj_detect_client("detect object"):
            stop()
            return True


def moveGoal(x,y):
    ac = actionlib.SimpleActionClient("move_base",MoveBaseAction)

    while(not ac.wait_for_server(rospy.Duration.from_sec(5.0))):
        rospy.loginfo("Waiting for server...\n")
    
    goal = MoveBaseGoal()

    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    #move towards goal
    goal.target_pose.pose.position = Point(x,y,0)
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = 0.0
    goal.target_pose.pose.orientation.w = 1.0

    rospy.loginfo("sending goal..\n")

    ac.send_goal(goal)

    print(ac.get_state())
    ac.wait_for_result(rospy.Duration(60))
  

    if(ac.get_state()==GoalStatus.ABORTED or ac.get_state()==GoalStatus.SUCCEEDED):
        rospy.loginfo("Reached!...\n")
        return True
    else:
        rospy.loginfo("failed\n")
        return False

def speechCommand():
    r = asr.Recognizer()
    with asr.Microphone() as src:
        print("speak!...robot is listening!!\n")
        audio = r.listen(src)
        print("here")
        text = None
        try:
            print("recognizing...please wait!!\n")
            text = r.recognize_google(audio)
            print("Your command: \n", text)
        except:
            print("error\ntext = speechCommand()")

    
    return text


def preProcesstext(text):
    task = {"objects":-1,"location":-1}
    for word in OBJECTS:
        if (text.lower().find(word)>0):
            task["objects"] =  word
    for word in LOCATIONS:
        if (text.lower().find(word)>0):
            task["location"] =  word
    return task

def processCommand(task):
    if (task["objects"]== -1 or task["location"] == -1):
        rospy.loginfo("Sorry could not understane, speak again!\n")
    goal_x,goal_y = COORDINATES[task["location"]]
    return moveGoal(goal_x,goal_y)

if __name__ == "__main__":
    rospy.init_node('speech_nagivation',anonymous=True)
    velocity_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
    img_sub = rospy.Subscriber("/ILRobot/ILRobotcamera/image_raw",Image,img_callback)
    rate = rospy.Rate(1)
    text = speechCommand()
    task = preProcesstext(text)
    success = processCommand(task)
    print(success)
    rospy.spin()

