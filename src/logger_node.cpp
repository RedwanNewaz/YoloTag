#include <ros/ros.h>
#include <memory>
#include <nav_msgs/Odometry.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Quaternion.h>
#include <std_srvs/Empty.h>
#include "logger_csv.h"

class TopicListener{
public:
    TopicListener()
    {
        int loggingFQ; 
        std::string topicName, outputPath; 
        nh_.param<int>("frequency", loggingFQ, 10);
        nh_.param<std::string>("topic", topicName, "/yolotag/state/filtered");
        nh_.param<std::string>("output", outputPath, "/home/roboticslab/yolo_ws/src/yolo_tag_detector/results/yolotag");

        std::vector<std::string>Header{"x", "y", "z", "yaw"};

        logger_ = std::make_unique<LoggerCSV>(Header, loggingFQ);
        state_sub_ = nh_.subscribe(topicName, 10, &TopicListener::state_callback, this);
        service_ = nh_.advertiseService("save_csv", &TopicListener::save_csv, this);
        logger_->set_out_folder(outputPath);
        fileSaved_ = false; 
        
    }

    ~TopicListener()
    {
        if(!fileSaved_)
        {
            ROS_INFO("[+] [TopicListener] saving csv file");
            logger_->save();
        }    
    }
protected:
    
    bool save_csv(std_srvs::Empty::Request &req, std_srvs::Empty::Response& res)
    {
        ROS_INFO("[+] [TopicListener] saving csv file");
        logger_->save();
        fileSaved_ = true; 

    }
    void state_callback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        auto pos = msg->pose.pose.position; 
        auto rot = msg->pose.pose.orientation;
        tf::Quaternion q(rot.x, rot.y, rot.z, rot.w);
        tf::Matrix3x3 m(q); 
        double roll, pitch, yaw; 
        m.getRPY(roll, pitch, yaw); 
        std::vector<double>state{pos.x, pos.y, pos.z, yaw};

        logger_->addRow(state);
    }
private:
    std::unique_ptr<LoggerCSV> logger_; 
    ros::NodeHandle nh_; 
    ros::Subscriber state_sub_; 
    ros::ServiceServer service_;
    bool fileSaved_; 
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "logger_node");
    TopicListener listener; 
    ros::spin(); 
    return 0; 
}