#include <iostream>
#include "Node/NodeHandlerWithStatic.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void ExtractPoint2D(char* filename, vector<Point2f> &extracted_point2d, int &image_width, int &image_height, float &arg_min_point_x, float& arg_min_point_y)
{
  extracted_point2d.clear();//기존에 들어있는 점을 지웁니다.
  ifstream poseFile;
  poseFile.open(filename);
  cout<<filename<<endl;
  if(poseFile.is_open())
  {
    while(!poseFile.eof())
    {
      string mat_pose;
      float p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12;
      
      getline(poseFile,mat_pose);
      sscanf(mat_pose.c_str(),"%e %e %e %e %e %e %e %e %e %e %e %e",&p1,&p2,&p3,&p4,&p5,&p6,&p7,&p8,&p9,&p10,&p11,&p12);
      Point2f sample_point = Point2f(p4,p12);//x,y좌표이다.
      extracted_point2d.push_back(sample_point);
    }
  }

  float max_point_x = -10000;
  float min_point_x = 10000;
  float max_point_y = -10000;
  float min_point_y = 10000;
  for(auto point = extracted_point2d.begin(); point != extracted_point2d.end(); point++)
  {
      if(point->x > max_point_x)
      {
        max_point_x = point->x;
      }
      if(point->y > max_point_y)
      {
        max_point_y = point->y;
      }
      if(point->x < min_point_x)
      {
        min_point_x = point->x;
      }
      if(point->y < min_point_y)
      {
        min_point_y = point->y;
      }
  }
  cout <<" minimum x : "<<min_point_x <<", minimum y : "<<min_point_y<<", maximum x : "<<max_point_x <<", maximum y : "<<max_point_y<<endl;
  image_width = (int)(max_point_x- min_point_x+20);//좌우로 10칸씩 추가하였다. 
  image_height = (int)(max_point_y - min_point_y)*10+20;//좌우로 10칸씩 추가하였다. 행은 10배로 늘렸다.
  arg_min_point_x = min_point_x;
  arg_min_point_y = min_point_y;
}

int main(int argc, char** argv)
{
  char filename[] = "/media/jeon/T7/Kitti dataset/data_odometry_poses/dataset/poses/00.txt";
  vector<Point2f> point2d;
  int image_width;
  int image_height;
  float min_point_x;
  float min_point_y;
  ExtractPoint2D(filename,point2d, image_width, image_height, min_point_x,min_point_y);//file 정보 및 이미지의 높이를 가져온다.
  vector<cv::String> file_names;
  glob("/media/jeon/T7/Kitti dataset/data_odometry_gray/dataset/sequences/00/image_0/*.png", file_names, false);
  NodeHandler nodehandler(3,30);
  nodehandler.Set_InstricParam(718.856,718.856,0.00,607.1928,185.2157);
  Mat map_image = Mat(image_height,image_width,CV_8UC3,Scalar(255,255,255));
  Vec3b* data = (Vec3b*)map_image.data;
  for(int camera_ind=0; camera_ind<file_names.size(); camera_ind++)
  {
    Mat candidate_image = imread(file_names[camera_ind],IMREAD_GRAYSCALE);//전체 파일들을 불러옵니다.
    bool b_IsGoodKeyFrame = nodehandler.Is_GoodKeyFrame(candidate_image);//적절한 키프레임인지 검사를 합니다.

    int col = 10+(point2d[camera_ind].x-min_point_x);
    int row = 10+((point2d[camera_ind].y-min_point_y))*10;
    clock_t start = clock();
    if(b_IsGoodKeyFrame)
    {//키프레임을 삽입하고, 맵포인트를 등록합니다.
      nodehandler.Make_KeyFrame(candidate_image);
      circle(map_image,Point(col,row),3,Scalar(0,0,255),1,8,0);
      cout<< (double)(clock()-start)/CLOCKS_PER_SEC<<"초"<<endl;
    }
    else
    {
      data[row*map_image.cols + col] = Vec3b(0,255,0);
    }
    imshow("map image",map_image);
    
    waitKey(1);
  }
}