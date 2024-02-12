from typing import Any
from ultralytics import YOLO
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import time
from functools import wraps
from collections import defaultdict
import numpy as np 
from .RvizPub import RvizPub
from .LowpassFilter import LowpassFilter


def compute_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"FPS for {func.__name__}: { 1.0 / elapsed_time:.3f}")
        return result
    return wrapper

rotation = np.array([[-1.0000000,  0.0000000,  0.0000000],
                [0.0000000,  0.0000000,  -1.0000000],
                [-0.0000000, -1.0000000,  0.0000000] ])

class YoloDetector:
    def __init__(self, modelWeight, camera_matrix, dist_coeffs, landmarks, alpha, outlier_dist) -> None:
        # Load a model
        self.landmarks = landmarks
        self.camera_matrix = np.reshape(np.array(camera_matrix, dtype=np.float32), (3, 3)) 
        self.dist_coeffs = np.array(dist_coeffs)

        self.model = YOLO(modelWeight)  # pretrained YOLOv8n model
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/yolo_detections', Image, queue_size=10)
        
        self.global_tag_corners = self.convert_tags()
        self.detection_buffer = None
        
        self.viz = RvizPub(5)
        self.lpf = LowpassFilter(alpha, outlier_dist)
        rospy.loginfo(f"[+] model weight | camera matrix {self.camera_matrix }")

    def decode_results(self, image, results):        
        
        self.detection_buffer = defaultdict(list)
        for result in results:
            for box in result.boxes:
            
                cords = box.xyxy[0].cpu().tolist()
                class_id = int(box.cls[0].cpu().item())
                conf = box.conf[0].cpu().item()
                
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, cords)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"{int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                box_cv = [x1, y1, x2, y2]
                self.detection_buffer[int(class_id)].append(box_cv)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image = self.detect(cv_image)
        image_message = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        self.pub.publish(image_message)

    def convert_tags(self):
        tags = {}

        for key, val in self.landmarks.items():
            tagSize = 0.2 if int(key) < 50 else 0.1 
            cx, cy, cz = val
            d = tagSize / 2.0
            x1, x2 = cx - d, cx + d
            y1, y2 = cy - 0, cy + 0  # because it is a flat paper tag 
            z1, z2 = cz - d, cz + d  

            left_bottom = rotation @ np.array([x1, y2, z1]).T
            right_bottom = rotation @ np.array([x2, y2, z1]).T 
            right_top = rotation @ np.array([[x2, y1, z2]]).T
            left_top = rotation @ np.array([[x1, y1, z2]]).T

            dst_points = np.array([ np.squeeze(left_bottom), np.squeeze(right_bottom), 
                                   np.squeeze(right_top), np.squeeze(left_top)], dtype=np.float32)
            tags[int(key)] = dst_points
        return tags



    def compute_coordinate(self):

        
        srcs = np.zeros((0, 2),dtype=np.float32) # pixel domain 
        dests = np.zeros((0, 3), dtype=np.float32) # metric domain
        for label in self.detection_buffer:
            for box in self.detection_buffer[label]:
                x1, y1, x2, y2 = box

                left_bottom = [x1, y2]
                right_bottom = [x2, y2] 
                right_top = [x2, y1]
                left_top = [x1, y1]

                src_points = np.array([ left_bottom, right_bottom, right_top, left_top], dtype=np.float32)
                dst_points = self.global_tag_corners[label]
                srcs = np.vstack((srcs, src_points))
                dests = np.vstack((dests, dst_points))
        
        if len(srcs) > 1:
            # print('src_points', srcs.shape)
            # print('dst_points', dests.shape)
            homography = self.computeHomographyMatrix(srcs, dests)
            tagArray = np.array(list(self.landmarks.values()))
            global_to_center = tagArray.mean(axis=0).T 
            C_R_O = homography[:3, :3]
            C_T_O = homography[:3, -1]

            transform = rotation @ C_T_O
            # transform = C_T_O
            # 7.5        1.28      3.0063
            # print('robot', np.squeeze(C_T_O), 'center_tf', np.squeeze(transform))
            zz = np.squeeze(transform)
            zz = self.lpf(zz)
            self.viz(zz.tolist())
                
        self.detection_buffer = None



    @compute_elapsed_time
    def detect(self, img):
        results = self.model(img.copy())  # return a list of Results objects
        
        # Process results list
        self.decode_results(img, results)
        if not self.detection_buffer is None:
            self.compute_coordinate()
        return img
    
    def computeHomographyMatrix(self, src_points, dst_points):

        if len(src_points) < 4 or len(dst_points) < 4:
            return None
        # Compute the homography matrix using the PnP method
        retval, rvec, tvec = cv2.solvePnP(dst_points, src_points, self.camera_matrix, self.dist_coeffs)
        # retval, rvec, tvec, _ = cv2.solvePnPRansac(dst_points, src_points, self.camera_matrix, self.dist_coeffs)
        # print(rvec, tvec)

        # Compute the rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)

        # Construct the homography matrix
        homography = np.concatenate((rot_matrix, tvec), axis=1)

        # Print the computed homography matrix
        # print("Homography Matrix:")
        # print(homography)
        return homography