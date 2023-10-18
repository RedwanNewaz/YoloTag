//
// Created by airlab on 10/18/23.
//
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include "YOLOv8AnnotationGenerator.h"
#include "DatasetGenerator.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
int main(int argc, char** argv) {

    ros::init(argc, argv, "dataset_generator");
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "apriltag dataset generator help message")
            ("bag", po::value<std::string>()->default_value("/home/airlab/catkin_ws/src/apriltag_dataset_generator/data/exp7.bag"), "find bag file")
            ("conf", po::value<std::string>()->default_value("/home/airlab/catkin_ws/src/apriltag_dataset_generator/config/param.yaml"), "apriltag conf file")
            ("output", po::value<std::string>()->default_value("/home/airlab/catkin_ws/src/apriltag_dataset_generator/data/bebop2"), "apriltag conf file")
            ("view", po::value<bool>()->default_value(false), "view images in dataset")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") ) {
        std::cout << desc << "\n";
        return 0;
    }

    rosbag::Bag bag;
    if (vm.count("bag")) {
        bag.open(vm["bag"].as<std::string>(), rosbag::bagmode::Read);
    } else {
        std::cerr << "bag file not found !.\n";
        return 1;
    }

    std::string configPath = vm["conf"].as<std::string>();
    std::string output = vm["output"].as<std::string>();
    bool viewImg = vm["view"].as<bool>();

    if (!fs::is_directory(configPath))
    {
        std::cerr << "config file not found !.\n";
        return 1;
    }


    std::vector<std::string> topics;
    topics.push_back(std::string("/bebop/image_raw"));

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    airlab::DatasetGenerator dataset(configPath, output, viewImg);


    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
                    // Print message information
//                    std::cout << "Topic: " << m.getTopic() << std::endl;
//                    std::cout << "Timestamp: " << m.getTime() << std::endl;
//                    std::cout << "Type: " << m.getDataType() << std::endl;
                    if(m.getDataType() == "sensor_msgs/Image")
                    {
                        sensor_msgs::Image::ConstPtr image = m.instantiate<sensor_msgs::Image>();
                        if (image != NULL) {
                            dataset.processImage(image);
                        }
                    }
                }

    bag.close();
    return 0;
}