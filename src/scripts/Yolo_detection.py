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

    img_height = cv_img.shape[0]
    img_width = cv_img.shape[1]

    # convert to blob to pass into model
    img_blob = cv2.dnn.blobFromImage(cv_img, 0.003922, (416, 416), swapRB=True, crop=False)
    #recommended by yolo authors, scale factor is 0.003922=1/255, width,height of blob is 320,320
    #accepted sizes are 320×320,416×416,608×608. More size means more accuracy but less speed

    # set of 80 class labels 
    class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

    #Declare List of colors as an array
    #Green, Blue, Red, cyan, yellow, purple
    #Split based on ',' and for every split, change type to int
    #convert that to a numpy array to apply color mask to the image numpy array
    class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors,(16,1))

    # Loading pretrained model 
    # input preprocessed blob into model and pass through the model
    # obtain the detection predictions by the model using forward() method
    yolo_model = cv2.dnn.readNetFromDarknet('/home/aditya/Capstone/ImaginationLearning/src/ILRobot/src/scripts/YoloWeights/yolov4.cfg',
                                            '/home/aditya/Capstone/ImaginationLearning/src/ILRobot/src/scripts/YoloWeights/yolov4.weights')

    # Get all layers from the yolo network
    # Loop and find the last layer (output layer) of the yolo network 
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    # input preprocessed blob into model and pass through the model
    yolo_model.setInput(img_blob)
    # obtain the detection layers by forwarding through till the output layer
    obj_detection_layers = yolo_model.forward(yolo_output_layer)


    ############## NMS Change 1 ###############
    # initialization for non-max suppression (NMS)
    # declare list for [class id], [box center, width & height[], [confidences]
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    ############## NMS Change 1 END ###########


    # loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
        # loop over the detections
        for object_detection in object_detection_layer:
            
            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]
        
            # take only predictions with confidence more than 50%
            if prediction_confidence > 0.50:

                #obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                
                
                ############## NMS Change 2 ###############
                #save class id, start x, y, width & height, confidences in a list for nms processing
                #make sure to pass confidence as float and width and height as integers
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                ############## NMS Change 2 END ###########


    ############## NMS Change 3 ###############
    # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
    # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    for max_valueid in max_value_ids:
        max_class_id = max_valueid[0]
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        
        #get the predicted class id and label
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
    ############## NMS Change 3 END ###########

        
        #obtain the bounding box end co-oridnates
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height
        
        #get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]
        
        #convert the color numpy array as a list and apply to text and box
        box_color = [int(c) for c in box_color]
        
        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))
        
        # draw rectangle and text in the image
        cv2.rectangle(cv_img, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(cv_img, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    cv2.imshow("Detection Output", cv_img)
    cv2.waitKey(200)
def main():
    rospy.init_node('Yolo_Detection',anonymous=True)
    img_sub = rospy.Subscriber("/ILRobot/ILRobotcamera/image_raw",Image,YoloProcessCallBack)


    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()









