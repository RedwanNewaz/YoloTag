#pragma once 

#include <iostream>
#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/Image.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include<opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>

// clone this repo to use json libary https://github.com/nlohmann/json

using json = nlohmann::json;


class SensorBuffer{
public:
SensorBuffer()
{
    _msgCount = 0;
}

~SensorBuffer()
{
    ROS_INFO("saving results");
    std::ofstream f("gt_pose.json");
    json j(_pose_map);
    std::cout << j.dump(4) << std::endl;
    // write prettified JSON to another file
    f << std::setw(4) << j << std::endl;
}

void operator()(const geometry_msgs::TransformStamped::ConstPtr& msg)
{
    if(_msgCount == 0)
        return; 
    
    ROS_INFO("vicon pose updated");
    std::vector<double> translation{
        msg->transform.translation.x,
        msg->transform.translation.y,
        msg->transform.translation.z
    };
    std::vector<double> rotation{
        msg->transform.rotation.x,
        msg->transform.rotation.y,
        msg->transform.rotation.z,
        msg->transform.rotation.w
    };

    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << _msgCount << ".png";

    _pose_map[ss.str()]["translation"] = translation;
    _pose_map[ss.str()]["rotation"] = rotation;

}

void operator()(const sensor_msgs::Image::ConstPtr& msg)
{
    ROS_INFO("bebop image updated");
    cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    _msgCount++; 
}

private:
    int _msgCount; 
    std::map<std::string, std::map<std::string, std::vector<double>>> _pose_map;

};
