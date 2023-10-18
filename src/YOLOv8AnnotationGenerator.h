//
// Created by airlab on 10/18/23.
//

#ifndef APRILTAG_DATASET_GENERATOR_YOLOV8ANNOTATIONGENERATOR_H
#define APRILTAG_DATASET_GENERATOR_YOLOV8ANNOTATIONGENERATOR_H
#include <iostream>
#include <fstream>
#include <vector>

class YOLOv8AnnotationGenerator {
private:
    std::ofstream outputFile;
public:
    YOLOv8AnnotationGenerator(const std::string &filename) {
        outputFile.open(filename);
        if (!outputFile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
        }
    }
    ~YOLOv8AnnotationGenerator()
    {
        closeFile();
    }

    void addAnnotation(const std::vector<std::vector<double>> &annotations) {
        if (!outputFile.is_open()) {
            std::cerr << "File not open!" << std::endl;
            return;
        }

        for (const auto &annotation : annotations) {
            if (annotation.size() < 5) {
                std::cerr << "Invalid annotation format!" << std::endl;
                return;
            }
            int classId = static_cast<int>(annotation[0]);
            double x = annotation[1];
            double y = annotation[2];
            double width = annotation[3];
            double height = annotation[4];

            outputFile << classId << " " << x << " " << y << " " << width << " " << height << "\n";
        }
        outputFile << std::endl;
    }

    void closeFile() {
        if (outputFile.is_open()) {
            outputFile.close();
        }
    }
};


#endif //APRILTAG_DATASET_GENERATOR_YOLOV8ANNOTATIONGENERATOR_H
