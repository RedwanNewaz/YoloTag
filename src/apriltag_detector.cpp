#include "apriltag_detector.h"

namespace airlab
{
  ApriltagDetector::ApriltagDetector():_td(apriltag_detector_create())
  {
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_add_family(_td, tf);

  }

  void ApriltagDetector::init(const std::string& param_file)
  {
    DetectorBase::init(param_file);
    _td->quad_decimate = _quad_decimate;
    _td->quad_sigma = _quad_sigma;
    _td->nthreads = _nthreads;
    _td->debug = false;
    _td->refine_edges = _refine_edges;
  }


  void ApriltagDetector::detect(cv::Mat& image, int camIndex, std::vector<std::vector<double>>& results)
  {
    cv::Mat img_uint8;
    cv::cvtColor(image, img_uint8, cv::COLOR_BGR2GRAY);
    
    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};
    zarray_t* detections = apriltag_detector_detect(_td, &im);
    
    // compute pose with respect to camera  
    decode_results(detections, camIndex, results);
    
    // overlay detections on images 
    annotate_image(image, detections);

    apriltag_detections_destroy(detections);
  }

 
  void ApriltagDetector::annotate_image(cv::Mat& frame, zarray_t* detections)
  {

    // Draw detection outlines
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        cv::line(frame, cv::Point(det->p[0][0], det->p[0][1]),
                  cv::Point(det->p[1][0], det->p[1][1]),
                  cv::Scalar(0, 0xff, 0), 2);
        cv::line(frame, cv::Point(det->p[0][0], det->p[0][1]),
                  cv::Point(det->p[3][0], det->p[3][1]),
                  cv::Scalar(0, 0, 0xff), 2);
        cv::line(frame, cv::Point(det->p[1][0], det->p[1][1]),
                  cv::Point(det->p[2][0], det->p[2][1]),
                  cv::Scalar(0xff, 0, 0), 2);
        cv::line(frame, cv::Point(det->p[2][0], det->p[2][1]),
                  cv::Point(det->p[3][0], det->p[3][1]),
                  cv::Scalar(0xff, 0, 0), 2);

        stringstream ss;
        ss << det->id;
        String text = ss.str();
        int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 1.0;
        int baseline;
        cv::Size textsize = cv::getTextSize(text, fontface, fontscale, 2,
                                        &baseline);
        cv::putText(frame, text, cv::Point(det->c[0]-textsize.width/2,
                                    det->c[1]+textsize.height/2),
                fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
    }

  }

  void ApriltagDetector::decode_results(zarray_t* detections, int camIndex, std::vector<std::vector<double>>& results)
  {
      //TODO send coord
      for (int i = 0; i < zarray_size(detections); i++)
      {
          apriltag_detection_t *det;
          zarray_get(detections, i, &det);
          if(det->hamming != 0)
              continue;

          double xmin, ymin, xmax, ymax;
          xmin = xmax = det->p[0][0];
          ymin = ymax = det->p[0][1];

          for (int j = 0; j < 4; ++j) {
              xmin = std::min(xmin, det->p[j][0]);
              ymin = std::min(ymin, det->p[j][1]);
              xmax = std::max(xmin, det->p[j][0]);
              ymax = std::max(xmin, det->p[j][1]);
          }

          double x = (xmin + xmax) / 2;
          double y = (ymin + ymax) / 2;
          double width = xmax - xmin;
          double height = ymax - ymin;
          int myInt = det->id;
          double index = static_cast<double>(myInt);
          results.push_back(std::vector<double>{index, x, y, width, height});

      }




  }

}