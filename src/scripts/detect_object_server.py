#!/home/aditya/Capstone/ImaginationLearning/ros_python3_ws/ros_env_py3/bin/python
import rospy
from ILRobot.srv import DetectObject
from ILRobot.srv import DetectObjectRequest
from ILRobot.srv import DetectObjectResponse
import cv2
import scipy
import numpy as np
from geometry_msgs.msg import Twist
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError

from sensor_msgs.msg import LaserScan



(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

bridge = CvBridge()
cv_img = None
PI = 3.141592653589793238
i=0
THRESH = 0.5


def updateXY():
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 20)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and filter for ROI
    image_number = 0
    max=0
    max_index=0

    if len(cnts)>0:
        stop()
        dontStop=False

    for i,c in enumerate(cnts):
        if(cv2.contourArea(c)<350):
            continue
        x,y,w,h = cv2.boundingRect(c)


def mostCommonColor(ar):
    NUM_CLUSTERS = 10

    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    return colour


def processImg(cv_img,speed):
    rospy.loginfo("processing Image")
    global i
    i+=1
    dontStop=True
    orig_img=cv_img.copy()
    height,width,_ = orig_img.shape
    hsv=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)

    brown_lo=np.array([0,0,0])
    brown_hi=np.array([105,220,220])

    mask=cv2.inRange(hsv,brown_lo,brown_hi)

    cv_img[mask>0]=(255,255,255)
    brown_lo=np.array([72,72,72])
    brown_hi=np.array([255,200,200])

    mask=cv2.inRange(cv_img,brown_lo,brown_hi)
    cv_img[mask>0]=(255,255,255)

    original = cv_img.copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 20)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and filter for ROI
    image_number = 0
    max=0
    max_index=0

    if len(cnts)>0:
        stop()
        dontStop=False

    for i,c in enumerate(cnts):
        if(cv2.contourArea(c)<350):
            continue
        x,y,w,h = cv2.boundingRect(c)
        one_bounding =cv_img.copy()
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (36,255,12), 2)
        
        ROI = original[y:y+h, x:x+w]
        new_image=ROI.copy()
        #if model(new_image)==banan:
        #    pass
        rospy.loginfo("geting bounding boxes!\n")
        #h=new_image[new_image<255]
        #cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        #image_number += 1
    if (dontStop!=True):
        dontStop = True
        rotate(speed)
    
    cv2.imshow("frame",orig_img)
    cv2.waitKey(5)

def stop():
    vel_twist = Twist()
    vel_twist.linear.x = 0
    vel_twist.linear.y = 0
    vel_twist.linear.z = 0

    vel_twist.angular.x = 0
    vel_twist.angular.y = 0
    vel_twist.angular.z = 0

    velocity_pub.publish(vel_twist)
    rospy.loginfo("Robot has stopped!")

def get_front_laser():
    rate.sleep()
    return laser_msg.ranges[360]


def laser_callback(msg):
    laser_msg = msg


#def trackingUpdate(one_bounding,tracker):
 #   
  #  return bbox

#def trackingInit(one_bounding,x,y,w,h):
#    return tracker

def validate_and_reach(one_bounding,x,y,w,h,height,width):
    
    rospy.loginfo("validating and moving towards !\n")

    tracker = cv2.TrackerCSRT_create()
    #tracker = cv2.TrackerKCF_create()
    #tracker = cv2.TrackerTLD_create()
    #tracker = cv2.TrackerMedianFlow_create()
    #tracker = cv2.TrackerMOSSE.create()
    
    bbox = (x, y, w, h)
    print("here before bounding box")
    tracker.init(one_bounding, bbox)  

    print("here after bounding box")
    
    #tracker = trackingInit(one_bounding,x,y,w,h)
    
    center_x,center_y = (x + w/2.0,y + h/2.0)
    img_dist = abs(width/2-center_x)
    while(img_dist>100):
        if (center_x<width):
            rotate(0.2,direction="anti-clock")
        
        if (center_x>width):
            rotate(0.2,direction = "clock")

        rospy.loginfo("before update\n")
    
        #bbox = trackingUpdate(x,y,w,h)
    
        _, bbox = tracker.update(one_bounding)
        rospy.loginfo("after update\n")
        img_dist = abs(width/2-(bbox[0]+bbox[2]/2))
        rospy.loginfo("distance.....{}".format(img_dist))
        cv2.imshow("frame2",one_bounding)
        cv2.waitKey(5)
    stop()
    dist = get_front_laser()
    while(dist>0.5):        
        print(dist)
        rospy.loginfo(dist)
        move(0.4)
        dist = get_front_laser()
        print()
    stop()
    cv2.destroyAllWindows()
    return True


def move(speed):
    vel_twist = Twist()

    vel_twist.linear.x = speed
    vel_twist.linear.y = 0
    vel_twist.linear.z = 0

    vel_twist.angular.x = 0
    vel_twist.angular.y = 0
    velocity_pub.publish(vel_twist)
    

def img_callback(ros_img):

    print("got frame!")
    global bridge
    global cv_img
    try:
        cv_img = bridge.imgmsg_to_cv2(ros_img,"bgr8")
    except CvBridgeError as e:
        print(e)


def degree2radian(degree):
    return degree*(PI/180.0)


def rotate(speed,direction="clock"):
    vel_twist = Twist()

    vel_twist.linear.x = 0
    vel_twist.linear.y = 0
    vel_twist.linear.z = 0

    vel_twist.angular.x = 0
    vel_twist.angular.y = 0
    if direction == "anti-clock":
        vel_twist.angular.z = speed

    if direction == "clock":
        vel_twist.angular.z = -speed
    velocity_pub.publish(vel_twist)


def srvHandler(req):
    speed = degree2radian(30)
    count = 0
    tic = 0
    if req.name == "detect object":
        rospy.loginfo("handling Service\n")
        while(tic<6.28/speed):
            processImg(cv_img,speed)
            rate.sleep()
            rotate(speed)
            tic+=1
    return True





def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )
    cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)






if __name__ == "__main__":
    
    rospy.init_node("Detect_Object_Search_Server")

    rate = rospy.Rate(1)
    img_sub = rospy.Subscriber("/ILRobot/ILRobotcamera/image_raw",Image,img_callback)
    velocity_pub = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
    laser_msg=LaserScan()
    laser_subscriber = rospy.Subscriber('/ILRobot/laser/scan', LaserScan, laser_callback)
    srv = rospy.Service("Detect_Object_Search_Service",DetectObject,srvHandler)
    
    print("Server is ready!!\n")
    rospy.spin()
    

