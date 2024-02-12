from typing import Any
import rospy
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np 
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry


class RvizPub:
    def __init__(self) -> None:
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.state_pub = rospy.Publisher('yolotag/state', Odometry, queue_size=10 )
        self.rviz_sub = rospy.Subscriber('/yolotag/state/filtered', Odometry, self.rviz_callback, queue_size=10)
        self.count = 0
        self.traj = []

    def __call__(self, pose) -> Any:
        
        quat = self.rotate(pose)
        
        msg = Pose()
        msg.position.x = pose[0] 
        msg.position.y = pose[1] * 3.5
        msg.position.z = pose[2] 
        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        self.publish_odom(msg)
        
        
        # self.count += 1 
        # self.traj.append(pose[:3])
        # self.publish_pose(pose)
    def rviz_callback(self, msg:Odometry):
        self.traj.append(msg.pose.pose.position)
        self.publish_pose(msg.pose.pose)        


    def rotate(self, pose):
        # sxyz = [pose[0], pose[1], pose[2], pose[3]]
        euler = np.array(euler_from_quaternion(pose))
        # print(euler)

        # rMat = np.array([
        #     [ -1.0000000,  0.0000000, -0.0000000],
        #     [  0.0000000,  0.0000000, -1.0000000],
        #     [  0.0000000, -1.0000000, -0.0000000 ]
        # ])

        # transform = rMat @ euler
        quat = quaternion_from_euler(0,  0 ,  -euler[2] )

        return quat 
    

    def publish_traj(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.id = 1
        marker.pose.orientation.w = 1

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.g = 0.0
        marker.color.r = marker.color.b = 1.0
        marker.color.a = 0.8  # Don't forget to set the alpha!

        for item in self.traj:
            # p = Point()
            # p.x = item[0]
            # p.y = item[1]
            # p.z = item[2]
            marker.points.append(item)
        
        self.marker_pub.publish(marker)


    def publish_odom(self, pose):
        msg = Odometry()
        msg.pose.pose = pose 
        msg.header.frame_id = "camera_link"
        msg.header.stamp = rospy.Time.now()
        self.state_pub.publish(msg)     





    
    def publish_pose(self, pose):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = "package://bebop2_controller/config/bebop.dae"
        marker.id = 0
        marker.pose = pose

        sxyz = [ marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w]
        euler = np.array(euler_from_quaternion(sxyz))
        dgain = 0.25
        dtheta = dgain * (2.0 - pose.position.x)
        quat = quaternion_from_euler(0, 0, euler[2] + dtheta)

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        # marker.scale.x = 0.25
        # marker.scale.y = 0.25
        # marker.scale.z = 0.1
        
        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001


        marker.color.g = 1.0
        marker.color.r = marker.color.b = 0.0
        marker.color.a = 0.8  # Don't forget to set the alpha!

        
        self.marker_pub.publish(marker)
        self.publish_traj()


