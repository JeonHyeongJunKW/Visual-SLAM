#include "Node/KeyFrame.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
/*
사진데이터를 읽어와야합니다. 한장씩 읽어옵니다. 
*/

int main(int argc, char** argv)
{
  vector<cv::String> file_names;
  glob("/media/jeon/T7/Kitti dataset/data_odometry_gray/dataset/sequences/00/image_0/*.png", file_names, false);
  for(int camera_ind=0; camera_ind<file_names.size(); camera_ind++)
  {
    Mat candidate_image = imread(file_names[camera_ind],IMREAD_GRAYSCALE);
    // imshow("hi ",candidate_image);
    
    while(waitKey(0) != 'n')
      continue;
  }
}