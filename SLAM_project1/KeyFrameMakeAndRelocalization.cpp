#include <iostream>
#include "Node/NodeHandler.h"
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
      Point2f sample_point = Point2f(p4,p8);//x,y좌표이다.
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
  image_width = (int)((max_point_x- min_point_x)*3+20);//좌우로 10칸씩 추가하였다. 
  image_height = (int)(max_point_y - min_point_y)*30+20;//좌우로 10칸씩 추가하였다. 행은 10배로 늘렸다.
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
  NodeHandler nodehandler(7,30);
  nodehandler.Set_InstricParam(718.856,718.856,0.00,607.1928,185.2157);

  Mat map_image;
  
  vector<int> key_index;
  for(int camera_ind=0; camera_ind<file_names.size(); camera_ind++)
  {
    Mat candidate_image = imread(file_names[camera_ind],IMREAD_GRAYSCALE);//전체 파일들을 불러옵니다.
    clock_t start = clock();
    bool b_IsGoodKeyFrame = nodehandler.ValidateAndAddFrame(candidate_image);//적절한 키프레임인지 검사를 합니다.

    int col = 10+(point2d[camera_ind].x-min_point_x)*3;
    int row = 10+((point2d[camera_ind].y-min_point_y))*30;
    
    if(b_IsGoodKeyFrame)
    {//키프레임을 삽입하고, 맵포인트를 등록합니다.
      key_index.push_back(camera_ind);
      // circle(map_image,Point(col,row),3,Scalar(0,0,255),1,8,0);
      // cout<<"연산에 걸린 시간"<<(double)(clock()-start)/CLOCKS_PER_SEC<<"초"<<endl;
      // data[row*map_image.cols + col] = Vec3b(0,0,255);
    }
    // else
    // {
    //   data[row*map_image.cols + col] = Vec3b(0,255,0);
    // }
    map_image = Mat(image_height,image_width,CV_8UC3,Scalar(255,255,255));
    Vec3b* data = (Vec3b*)map_image.data;
    for(int j=0; j<camera_ind; j++)
    {
      int track_col = 10+(point2d[j].x-min_point_x)*3;
      int track_row = 10+((point2d[j].y-min_point_y))*30;
      data[track_row *map_image.cols + track_col] = Vec3b(0,255,0);
    }
    for(int k=0; k<key_index.size(); k++)
    {
      circle(map_image,Point(10+(point2d[key_index[k]].x-min_point_x)*3,10+((point2d[key_index[k]].y-min_point_y))*30),10,Scalar(0,0,255),1,8,0);
    }
    for(int l=0; l<nodehandler.Get_LocalKeyFrame().size(); l++)
    {
      int neighbor_keyframe = nodehandler.Get_LocalKeyFrame()[l]->Get_KeyIndex();
      circle(map_image,Point(10+(point2d[neighbor_keyframe].x-min_point_x)*3,10+((point2d[neighbor_keyframe].y-min_point_y))*30),10,Scalar(255,0,0),-1,8,0);
    }
    imshow("map image",map_image);
    waitKey(1);
    // while( waitKey(-1) !='n')
    // {
    //   continue;
    // }
  }
}