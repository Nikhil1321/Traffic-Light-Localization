#!/usr/bin/env python3

"""
Created on Tue April 26 2022
@author: Nikhil Raj & Ronak Satpute

Course: Automated and Connected Driving Challenges Research Project 2022S

Project: Cross-Modal Depth Estimation for Traffic Light Detection
Project Task: The task is to develop a methodology for estimating the 3D
              position of a traffic light, given a binary image segmentation
              mask and a lidar point cloud.

This is the main ROS Node executable file.
Description: This file excutes the complete program.

"""

import rospy

# ROS imports
import tf2_ros

# Python libraries
import numpy as np
from cv_bridge import CvBridge

# ROS message imports
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import message_filters

# Custom imports (code developed for this particular package)
from traffic_light_localization.msg import Position3dTraffic, SinglePosition
from Localization.Localization import Localization
from global_var import *


class TrafficLight3DPosition:
    '''
    The TrafficLight3DPosition object takes in combination of LIDAR data and
    Binary Segmented Mask. Performs time-synchronization of the incoming data
    stream and computes 3D Spatial Coordinates of the traffic lights.
    After computation the data is published as a ROS message.

    Parameters:

    Returns::

    '''


    def __init__(self):
        self.cloud_in = None
        self.cloud_out = None
        self.semseg_img_in = None

        # Projection Matrix will be updated during runtime if any change is
        # detected
        self.projection_matrix = PROJETION_MATRIX

        # Subscribers to Point Cloud, Binary Mask and Camera Info
        self.point_cloud_sub = message_filters.Subscriber(
                               POINT_CLOUD_SUB_TOPIC, PointCloud2)
        self.binary_mask_sub = message_filters.Subscriber(
                               BINARY_MASK_SUB_TOPIC, Image)
        self.cam_info_sub = message_filters.Subscriber(
                            CAMERA_INFO_SUB_TOPIC, CameraInfo)

        # Time Synchronizer and callback
        self.ts = message_filters.TimeSynchronizer([self.point_cloud_sub,
                  self.binary_mask_sub, self.cam_info_sub], 10)
        self.ts.registerCallback(self.callback)

        # TF Listener setup
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(1))
        self.tl = tf2_ros.TransformListener(self.tf_buffer)

    def callback(self, point_cloud, image, camera_info):
        '''
        Primary callback function, performs the transformation of point
        cloud data to camera frame and extracts the recorded spatial
        coordinates by LIDAR. Finally initializes localization class to
        find the spatial coordinates of the detected traffic ights in the
        image using driver function of localization class.

        Parameters:
            point_cloud (PointCloud2) : Raw Point cloud data (ROS Message)
            image (Image) : Raw Image data (ROS Message)
            camera_info (CameraInfo) : Raw Camera Info data (ROS Message)

        Returns:

        '''

        # setting up the variable
        self.cloud_in = point_cloud
        self.semseg_img_in = image

        # Check / Analysis Delete
        self.time = rospy.get_rostime()

        # Updating Projectioin Matrix if change is detected
        if set(tuple(self.projection_matrix)) != set(camera_info.P):
            self.projection_matrix = camera_info.P

        # Setting up the frames for transform
        source_frame = self.cloud_in.header.frame_id
        target_frame = self.semseg_img_in.header.frame_id

        # Finding the transformation matrix for performing frame transformation
        transformation_matrix = self.tf_buffer.lookup_transform(target_frame,
                                source_frame, rospy.Time(0))

        # Tranforming LIDAR data from LIDAR frame to Camera Frame
        self.cloud_out = do_transform_cloud(self.cloud_in,
                         transformation_matrix)

        # Initiate localisation method for starting the computation
        self.initiate_localization()

    def initiate_localization(self):
        '''
        Extracts Image and LIDAR spatial coordinates from ROS Image message
        and ROS PointCloud2 message. Instantiates Localization class and calls
        localization driver method for performing computation. Finally calls
        the publish method to publish the data to predefined topic for the use
        by other modules down the process pipeline.

        Parameters:

        Returns:

        '''

        # Extracting image from ROS Image message using CV Bridge.
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(self.semseg_img_in,
                   desired_encoding='passthrough')

        # Extracting spatial coordinates from ROS LIDAR message using python
        # generator and list
        generator = point_cloud2.read_points(self.cloud_out,
                    field_names=("x", "y", "z"), skip_nans=True)
        generator = list(generator)
        raw_pcd = np.array(generator)

        # Reshaping projection matrix
        projection_matrix = np.array(self.projection_matrix).reshape(3,4)

        # Instantiating Localization class and calling its driver method
        localize = Localization(raw_pcd, cv_image, projection_matrix)
        result = localize.localization_driver()

        # Calling the publish method
        self.publish_result(result)

    def publish_result(self, result):
        '''
        Publishes the computed result to a predefined topic.

        Parameters:
            result (list) : Final result containing the spatial co-ordinates
                            of each detected traffic light in a given image.

        Returns:

        '''

        # Calling the make message method to create the message according to
        # custom message format
        pub_msg = self.make_msg(result)

        # Creating a publisher object based on pre-defined topic name and
        # publishing the create message
        pub = rospy.Publisher(TOPIC_NAME, Position3dTraffic, queue_size=10)
        pub.publish(pub_msg)


    def make_msg(self, result):
        '''
        Creates the message according to the custom message format.

        Parameters:
            result (list) : Final result containing the spatial co-ordinates
                            of each detected traffic light in a given image.

        Returns:
            msg (Position3DTrafic.msg) : Custom defined ROS message.

        '''

        msg = Position3dTraffic()

        # Header
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.semseg_img_in.header.frame_id

        # Field Descriptor
        msg.fields = ['X Mean', 'X Max', 'X Min', 'Y Mean', 'Y Max', 'Y Min', 'Z Mean', 'Z Max', 'Z Min']

        # Actual sensor data acquisition time
        msg.sourcetime = self.semseg_img_in.header.stamp.to_sec()

        # Total No of traffic lights detected
        msg.count = len(result)

        # Position data for all traffic lights that has been resent in the image.
        # If row contains 30000.0 as all nine values, consider it out of sensor range
        # As no data from LIDAR was computed.
        for i in range(len(result)):
            val = SinglePosition()
            val.singlepos = result[i]

            msg.data.append(val)

        return msg


if __name__ == '__main__':
    # Initializing the node and printing the confirmation.
    rospy.init_node(NODE_NAME, anonymous = True)
    rospy.loginfo('Traffic Light Localization Node --> Started.')

    # Instatiating the Localization Class
    transform_point_cloud = TrafficLight3DPosition()

    rospy.spin()
