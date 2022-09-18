#!/usr/bin/env python3
"""
Created on Thu September 01 2022
@author: Nikhil Raj & Ronak Satpute

Listener Check Node
Description: This is the listener ROS Node for verifying the localization node working.

"""

import rospy

# Custom imports (code developed for this particular package)
from traffic_light_localization.msg import Position3dTraffic
from final_global_var import *


def callback(msg):
    rospy.loginfo('Message Recieved')

def subscriber():
    sub = rospy.Subscriber(TOPIC_NAME, Position3dTraffic, callback = callback)
    rospy.spin()

if __name__ == '__main__':

    rospy.init_node('Listener Check', anonymous = True)
    rospy.loginfo('Listener Check Node has been started.')

    subscriber()
