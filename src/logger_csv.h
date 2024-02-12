#pragma once 

#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <string>
#include <fstream>
#include <iomanip>

using namespace std::literals;

class LoggerCSV{

  using TIME_POINT = std::chrono::time_point<std::chrono::system_clock>;

public:
  // this is a default contructor without st.pop_back();header. 
  // one can manually add header file using addHeader function 
  LoggerCSV()
  {
    m_header_enabled = false; 
    m_timer_enabled = false;
    m_out_folder = "."; 
  }

  /// use this constructor to create a csv file with header 
  LoggerCSV(const std::vector<std::string>&header)
  {
    m_header_enabled = false; 
    m_timer_enabled = false; 
    m_out_folder = ".";
    addHeader(header);
  }

  /// use this contructor when you want to record data row at different frequency than sensor frequency   
  LoggerCSV(const std::vector<std::string>&header, int frequency)
  {
    m_header_enabled = false; 
    m_timer_enabled = false; 
    m_out_folder = ".";
    addHeader(header);
    // initiate last update time 1 min later to write data row
    // write time only depends on actual data not header
    m_last_update_time =  std::chrono::system_clock::now() - 1min;
    m_timer_enabled = true; 
    // calculate pause time in millisecond between two data rows 
    m_pause_time_ms = 1000 / frequency; 
    
  }
  void set_out_folder(const std::string& filepath)
  {
    m_out_folder = filepath;
  }

  void save()
  {
    // when this class will be distructed, we will write the logger values from ROM to a csv file on the hard drive. 
    std::string value = m_data.str();

    // don't write file if there is no data added to the logger 
    if(!value.length())
      return;

    // remove last new line from the string 
    value.pop_back(); 

    // make sure you have unique name for each logger file. Use system timestamp to serve this purpose 
    std::string filename = m_out_folder + "/logger_" + getTimestamp() + ".csv"; 
    std::ofstream myfile(filename);
    myfile << value;
    myfile.close(); 
  }
  
  ~LoggerCSV()
  {
 
  }

  /// this function is used to generate unique name for a file with timestamp 
  std::string getTimestamp()
  {
    const TIME_POINT now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
   
    std::stringstream ss; 
    ss << std::put_time(std::localtime(&t_c), "%F_%T"); 
    return ss.str(); 
  }
  
  void addHeader(const std::vector<std::string>&data)
  {
    addRow(data); 
    m_header_len = data.size();
    m_header_enabled = true; 
  }
  
  template<typename T>
  void addRow(const std::vector<T>&data)
  {
    if(m_timer_enabled)
    {
      // if timer is enabled then discard the data point based on the pause limit 
      const auto end = std::chrono::system_clock::now();
      int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_last_update_time).count();
      
      if(elapsed_time <= m_pause_time_ms)
        return;
      // std::cout << elapsed_time << " Paused time " << m_pause_time_ms << std::endl; 
      m_last_update_time =  std::chrono::system_clock::now(); 
    }

    // if header is present then check the number of columns match with header column 
    if(m_header_enabled)
      if(m_header_len != data.size())
      {
        std::cerr << "header len does not match with data len" << std::endl;  
        return;
      }

    // we are now ready to store the data into RAM using stringstream 
    int index = 0;
    for(const auto& item: data)
      if(++index < data.size())
        m_data << item << ","; 
      else
        m_data << item; 
    m_data << "\n"; 


  }

private:
  std::stringstream m_data; 
  size_t m_header_len;
  bool m_header_enabled; 
  bool m_timer_enabled; 
  TIME_POINT m_last_update_time; 
  int m_pause_time_ms;
  std::string m_out_folder; 

};
