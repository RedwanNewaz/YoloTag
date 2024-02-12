#!/usr/bin/python3
import rospy
from sensor_msgs.msg import Image
import numpy as np 
from YoloTag import YoloDetector

# tested /home/roboticslab/Storage/Old_bags/bebob2_bag/exp7.bag

def main():
    rospy.init_node('yolo_detector', anonymous=True)
    modelWeight = rospy.get_param('~model_weight')
    camera_matrix = rospy.get_param('~camera_matrix')
    dist_coeffs = rospy.get_param('~dist_coeffs')
    Tags = rospy.get_param('~Tags')
    order = rospy.get_param('~order')
    a_coeff = rospy.get_param('~a_coeff')
    b_coeff = rospy.get_param('~b_coeff')
    conf_thrs = rospy.get_param('~conf_thrs')
    rospy.loginfo(f"[+] YoloTag initializing with {modelWeight}")
    yolo = YoloDetector(modelWeight, camera_matrix, dist_coeffs, Tags, a_coeff, b_coeff, order, conf_thrs)
    rospy.Subscriber("/bebop/image_raw", Image, yolo.image_callback)

    rospy.spin()

if __name__ == '__main__':
    main()