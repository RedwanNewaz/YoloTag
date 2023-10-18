//
// Created by airlab on 10/18/23.
//

#ifndef APRILTAG_DATASET_GENERATOR_DATASETGENERATOR_H
#define APRILTAG_DATASET_GENERATOR_DATASETGENERATOR_H
#include <ros/ros.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>

#include "apriltag_detector.h"
#include "YOLOv8AnnotationGenerator.h"

namespace airlab {

    class DatasetGenerator {
    public:
        DatasetGenerator(const string &apriltagConfigPath, const string &datasetRootPath, bool viewImage = false);
        void processImage(const sensor_msgs::ImageConstPtr& msg);

    private:
        std::string apriltagConfigPath_, datasetRootPath_;
        int imgCounter_;
        bool viewImage_;
        airlab::ApriltagDetector detector_;
        std::pair<std::string, std::string> getFileNames() const;
        void checkDirs(const std::string& path);



    };

} // airlab

#endif //APRILTAG_DATASET_GENERATOR_DATASETGENERATOR_H
