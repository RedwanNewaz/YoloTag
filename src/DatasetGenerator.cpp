//
// Created by airlab on 10/18/23.
//

#include "DatasetGenerator.h"
#include <filesystem>
namespace fs = boost::filesystem;

namespace airlab {
    DatasetGenerator::DatasetGenerator(const string &apriltagConfigPath, const string &datasetRootPath, bool viewImage)
            : apriltagConfigPath_(apriltagConfigPath), datasetRootPath_(datasetRootPath),
            viewImage_(viewImage), imgCounter_(0)
    {
        detector_.init(apriltagConfigPath);
        checkDirs(datasetRootPath);
        auto imgFolder = datasetRootPath_ + "/img";
        auto annoFolder = datasetRootPath_ + "/annotation";
        checkDirs(imgFolder);
        checkDirs(annoFolder);


    }

    std::pair<std::string, std::string> DatasetGenerator::getFileNames()const {
        std::stringstream formattedString;
        formattedString << std::setw(4) << std::setfill('0') << imgCounter_;
        std::string basename = formattedString.str();
        std::string imgFile = datasetRootPath_ + "/img/" + basename + ".jpg";
        std::string txtFile = datasetRootPath_ + "/annotation/" + basename + ".txt";
        return std::make_pair(imgFile, txtFile);
    }

    void DatasetGenerator::processImage(const sensor_msgs::ImageConstPtr &msg) {
        cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        std::vector<std::vector<double>> results;
        cv::Mat originalImg = img.clone();
        detector_.detect(img, 0, results);

        if(!results.empty())
        {
            ++imgCounter_;
            auto name = getFileNames();
            cv::imwrite(name.first, originalImg);
            YOLOv8AnnotationGenerator annotator(name.second);
            annotator.addAnnotation(results);
            ROS_INFO("[ImageCallback] processing = %d", imgCounter_);
        }

        if(viewImage_)
        {
            cv::imshow("view", img);
            cv::waitKey(30);
        }

    }

    void DatasetGenerator::checkDirs(const string &path) {
        if (!fs::is_directory(path)) {
            if (fs::create_directory(path)) {
                std::cout << path << " created successfully." << std::endl;
            } else {
                std::cerr << "Failed to create the directory." << std::endl;
            }
        } else {
            std::cout << path <<" already exists." << std::endl;
        }
    }
} // airlab