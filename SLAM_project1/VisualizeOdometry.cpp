#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;
using namespace cv;

void VisualizeOdometry2D(const vector<Point2f> points)
{
  float max_point_x = -10000;
  float min_point_x = 10000;
  float max_point_y = -10000;
  float min_point_y = 10000;
  for(auto point = points.begin(); point != points.end(); point++)
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
  int image_width = (int)(max_point_x- min_point_x+20);//좌우로 10칸씩 추가하였다. 
  int image_height = (int)(max_point_y - min_point_y)*10+20;//좌우로 10칸씩 추가하였다. 행은 10배로 늘렸다.
  Mat map_image = Mat::zeros(image_height,image_width,CV_8UC3);
  
  Vec3b* data = (Vec3b*)map_image.data;
  cout<<"width : "<<image_width<<" height : "<< image_height<<endl;
  
  for(int i =0; i < points.size(); i++)
  {
    int col = 10+(points[i].x-min_point_x);
    int row = 10+((points[i].y-min_point_y))*10;
    data[row*map_image.cols + col] = Vec3b(0,0,255);
    imshow("map image",map_image);
    waitKey(10);
  }
}

void ExtractPoint2D(char* filename, vector<Point2f> &extracted_point2d)
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
}

int main(int argc, char** argv)
{
  /*
  example) 
  ./VisualizeOdometry "/media/jeon/T7/Kitti dataset/data_odometry_poses/dataset/poses/00.txt"
  */
  if(argc <2)
  {
    cout<<"the number of argument is " <<argc<<endl;
    return 0;
  }
  char* filename = argv[1];
  vector<Point2f> point2d;
  ExtractPoint2D(filename,point2d);
  VisualizeOdometry2D(point2d);
  return 0;
}