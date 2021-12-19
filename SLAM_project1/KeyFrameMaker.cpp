#include <iostream>
#include "Node/NodeHandler.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  vector<cv::String> file_names;
  glob("/media/jeon/T7/Kitti dataset/data_odometry_gray/dataset/sequences/00/image_0/*.png", file_names, false);
  NodeHandler nodehandler;
  for(int camera_ind=0; camera_ind<500; camera_ind++)
  {
    Mat candidate_image = imread(file_names[camera_ind],IMREAD_GRAYSCALE);//전체 파일들을 불러옵니다.
    bool test = nodehandler.Is_GoodKeyFrame(candidate_image);//적절한 키프레임인지 검사를 합니다.
  }
}