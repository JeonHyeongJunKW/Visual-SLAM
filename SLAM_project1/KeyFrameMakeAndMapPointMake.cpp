#include <iostream>
#include "Node/NodeHandler.h"
#include <opencv2/opencv.hpp>
// #include "DBoW2.h"
#include "Node/CameraTool.h"
#include "Tracking/Tracking.h"
#include "LocalMapping/LocalMapping.h"

#include <thread>
#include <mutex>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  NodeHandler nodehandler(1,30);
  nodehandler.Set_InstricParam(718.856,718.856,0.00,607.1928,185.2157);

  thread t_thread(tracking_thread,ref(nodehandler));
  thread l_thread(LocalMapping,ref(nodehandler));
  //--지도상에 추정치를 그리는부분 초기화--begin
  vector<Point3f> point3f;
  int image_width;
  int image_height;
  float min_point_x;
  float min_point_z;
  char filename[] = "/media/jeon/T7/Kitti dataset/data_odometry_poses/dataset/poses/00.txt";
  ExtractPoint3D(filename, point3f, image_width, image_height, min_point_x,min_point_z);//file 정보 및 이미지의 높이를 가져온다.
  Mat map_image= Mat(image_height,image_width,CV_8UC3,Scalar(255,255,255));
  Vec3b* data = (Vec3b*)map_image.data;
  int last_size =nodehandler.estimated_pose.size();
  while(true)
  {
    // //그리는부분 
    if(nodehandler.estimated_pose.size()-last_size>0)
    {
      nodehandler.m_sharedlock.lock();//뮤텍스락으로 키프레임에 대한 노드핸들링권한을 가져옵니다.
      Mat last_estimated_pose= nodehandler.estimated_pose.back();
      int origin_col = 10+(point3f[nodehandler.int_CurrentFrameIdx-1].x-min_point_x)*1;//실제 x좌표를 그림좌표 x로 바꾼것
      int origin_row = 10+((point3f[nodehandler.int_CurrentFrameIdx-1].z-min_point_z))*1;//실제 z좌표를 그림좌표 y로 바꾼것
      int estimated_col = 10+(last_estimated_pose.at<double>(0,3)-min_point_x)*1;//추정된 x좌표를 그림좌표로 바꾼것
      int estimated_row = 10+(last_estimated_pose.at<double>(2,3)-min_point_z)*1;//추정된 z좌표를 그림좌표로 바꾼것
      circle(map_image,Point(origin_col,origin_row),2,Scalar(0,0,255),1,8,0);//실제 좌표를 지도상에 그립니다.
      circle(map_image,Point(estimated_col,estimated_row),2,Scalar(255,0,0),1,8,0);//추정된 좌표를 지도상에 그립니다. 
      imshow("map image",map_image);//opencv상에 그립니다. 
      waitKey(1);
      last_size =nodehandler.estimated_pose.size();
      
      nodehandler.m_sharedlock.unlock();//뮤텍스락으로 노드핸들링권한을 반납하여, tracking node가 일하게합니다.
    }
    //그리는 부분 --end
  }


  t_thread.join();
  l_thread.join();
}