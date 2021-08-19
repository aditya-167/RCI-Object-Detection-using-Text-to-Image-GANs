#!/home/aditya/Capstone/ImaginationLearning/ros_python3_ws/ros_env_py3/bin/python

import logging
import threading

import rospy
from std_msgs.msg import String

from flask import Flask, render_template
from flask_ask import Ask, statement, question, session

threading.Thread(target=lambda: rospy.init_node('fetch_obj_node', disable_signals=True)).start()
pub = rospy.Publisher('test_pub', String, queue_size=1)

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

@ask.launch
def launch():
    welcome_msg = render_template('welcome')
    return question(welcome_msg)

@ask.intent('WelcomeIntent')
def hello(firstname):
    text = render_template('hello', firstname=firstname)
    pub.publish(firstname)
    #f=open("templates/user.txt",'w')
    #f.write(firstname)
    #f.close()
    return statement(text).simple_card('Hello', text)

@ask.intent('FetchIntent')
def fetch_sentence(object,location):
    #f=open('templates/user.txt','r')
    #user = f.read()
    #if (len(user) == 0):
    #    text = render_template('unauthorized')
    #    return statement(text).simple_card('Hello', text)
    text = render_template('fetch_sentence', object=object,location=location)
    pub.publish(object)
    pub.publish(location)
    print(object,location)
    return statement(text).simple_card('Hello', text)

if __name__ == '__main__':
    app.run(debug=True)