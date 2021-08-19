#!/home/aditya/Capstone/ImaginationLearning/ros_python3_ws/ros_env_py3/bin/python

"""

@author: aditya/bhart
"""

import numpy as np
import cv2
import rospy
from std_msgs.msg import String 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import imutils

import sys

bridge = CvBridge()

# load the image to detect, get width, height 
def YoloProcessCallBack(ros_img):
	print("reading frame")
	global bridge
	try:
		cv_img = bridge.imgmsg_to_cv2(ros_img,"bgr8")
	except CvBridgeError as e:
		print(e)
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor","banana","apple","orange"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe("/home/aditya/Capstone/ImaginationLearning/src/ILRobot/src/scripts/YoloWeights/MobileNetSSD_deploy.prototxt", 
	"/home/aditya/Capstone/ImaginationLearning/src/ILRobot/src/scripts/YoloWeights/MobileNetSSD_deploy.caffemodel")
	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	print("[INFO] starting video stream...")
	#vs = VideoStream(src=0).start()
	# loop over the frames from the video stream
	#while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		#frame = vs.read()
	frame = imutils.resize(cv_img, width=400)
		# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		# pass the blob through the network and obtain the detections and
		# predictions
	net.setInput(blob)
	detections = net.forward()

		# loop over the detections
	for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
		confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
		if confidence > 0.2:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, "object", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

						# show the output frame
	cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
	cv2.imshow("Frame", frame)
	cv2.waitKey(20)
		# if the `q` key was pressed, break from the loop
		# update the FPS counter
		# stop the timer and display FPS information
	
	# do a bit of cleanup


def main():
	rospy.init_node('MobileNet',anonymous=True)
	
	img_sub = rospy.Subscriber("/ILRobot/ILRobotcamera/image_raw",Image,YoloProcessCallBack)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()









