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
  nodehandler.Set_InstricParam(718.856,718.856,0.00,607.1928,185.2157);
  for(int camera_ind=0; camera_ind<file_names.size(); camera_ind++)
  {
    Mat candidate_image = imread(file_names[camera_ind],IMREAD_GRAYSCALE);//전체 파일들을 불러옵니다.
    bool b_IsGoodKeyFrame = nodehandler.Is_GoodKeyFrame(candidate_image);//적절한 키프레임인지 검사를 합니다.

    if(b_IsGoodKeyFrame)
    {//키프레임을 삽입하고, 맵포인트를 등록합니다.
      nodehandler.Make_KeyFrame(candidate_image);
    }
  }
}