#pragma once 

#include <iostream>
#include<opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <tf2/LinearMath/Transform.h>

using namespace std;
using namespace cv;

namespace airlab{
    inline void printTransformation(const tf2::Transform& transform)
    {
        std::cout << "Translation: (" << transform.getOrigin().x()
                << ", " << transform.getOrigin().y()
                << ", " << transform.getOrigin().z() << ")" << std::endl;
        std::cout << "Rotation: (" << transform.getRotation().x()
                << ", " << transform.getRotation().y()
                << ", " << transform.getRotation().z()
                << ", " << transform.getRotation().w() << ")" << std::endl;
    }

    class DetectorBase{

    public:
      DetectorBase()
      {
        // init("airlab_cams.yaml");
        
      }

      void init(const std::string& paramFile)
      {
        std::string package_name = "multicam_tag_state_estimator";
        auto config_path = paramFile;
        _node_conf = YAML::LoadFile(config_path);
        readApriltagParams();

      }

      

    protected:      
        std::vector<YAML::Node> _camConfs;
        std::vector<tf2::Transform> _mapTfs;
        std::vector<std::string> _frames; 
        YAML::Node _node_conf; 

    protected:
        double _quad_decimate;
        double _quad_sigma; 
        int _nthreads; 
        bool _refine_edges;
        bool _calibration;     

    protected:
        virtual void detect(cv::Mat& img_uint8, int camIndex, std::vector<std::vector<double>>& results) = 0;




    private:    

        void readApriltagParams()
        {
            _quad_decimate = _node_conf["quad_decimate"].as<double>();
            _quad_sigma = _node_conf["quad_sigma"].as<double>();
            _nthreads = _node_conf["nthreads"].as<int>();
            _refine_edges = _node_conf["refine_edges"].as<bool>();

        }    
        


        tf2::Transform convertTFfromVector(const std::vector<double>& pose)
        {
   
            tf2::Transform transform; 

            tf2::Vector3 origin(pose.at(0), pose.at(1), pose.at(2));
            tf2::Quaternion rotation(pose.at(3), pose.at(4), pose.at(5), pose.at(6)); 

            transform.setOrigin(origin);
            transform.setRotation(rotation); 

            return transform; 
        }

        tf2::Transform getMapCoord(const tf2::Transform& pose, const tf2::Transform& C_T_M)
        {
            // pose is C_T_R : camera to robot transformation 
            // C_T_M : camera to map transformation
            // need to find M_T_R transformation 
            tf2::Transform M_T_R = C_T_M.inverseTimes(pose);
            
            // fix yaw angle 
            auto q = M_T_R.getRotation(); 
            tf2::Matrix3x3 m(q); 
            double roll, pitch, yaw; 
            m.getRPY(roll, pitch, yaw); 
            q.setRPY(0, 0, yaw + M_PI);
            M_T_R.setRotation(q);

            return M_T_R; 
        }  
     
  };
}