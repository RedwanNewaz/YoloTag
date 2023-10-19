#pragma once 

#include <iostream>
#include "detector_base.h"
#include <apriltag.h>
#include <tag36h11.h>
#include <unordered_set>


namespace airlab{
  class ApriltagDetector : public DetectorBase
  {
  public:
    ApriltagDetector();
    void init(const std::string& param_file);

    void detect(cv::Mat& image, int camIndex, std::vector<std::vector<double>>& results) override;

  
  private:
    apriltag_detector_t* const _td;

    void decode_results(zarray_t* detections, int camIndex, std::vector<std::vector<double>>& results);

    void annotate_image(cv::Mat& image, zarray_t* detections);

    

  };
}