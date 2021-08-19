#!/home/aditya/Capstone/ImaginationLearning/ros_python3_ws/ros_env_py3/bin/python
import cv2
#tracker = cv2.TrackerBoosting_create()

"""
import cv2
import numpy as np

def imageRead(image_name):

    img_path = "../Img/"
    print("reading Image....\n")

    img = cv2.imread(img_path+image_name)
    cv2.namedWindow("Banana Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Banana Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def readStream(file=None,video_file=False):
    if video_file:
        video_path = "../Videos/"
        print("reading Video file...\n")
        capture = cv2.VideoCapture(video_path+file)

        while(True):
            _,frame = capture.read()
            cv2.imshow("Video_frame",frame)

            if cv2.waitKey(10 & 0xFF == ord('q')):
                break
        capture.release()
        cv2.destroyAllWindows()
    else:
        video_path = 0
        print("reading webcam streame...\n")
        capture = cv2.VideoCapture(video_path)

        while(True):
            _,frame = capture.read()
            cv2.imshow("Video_frame",frame)

            if (cv2.waitKey(10) & 0xFF == ord('q')):
                break

        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img_name = "Banana.jpeg"
    imageRead(img_name)
    readStream()
"""