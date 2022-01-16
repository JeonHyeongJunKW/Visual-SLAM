#include "CameraTool.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <thread>
using namespace std;
using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void Custom_undisortionPoints(vector<Point2d> arg_InputPoints, Mat IntrinsicParam,vector<Point2d> &arg_OuputPoints)
{
    int N = arg_InputPoints.size();
    vector<Point3d> homoInputPoints;
    convertPointsToHomogeneous(arg_InputPoints,homoInputPoints);
    Mat inv_param = IntrinsicParam.inv();
    vector<Point3d> homoOutputPoints;
    for(int i=0; i< N; i++)
    {
        Mat tempMat(homoInputPoints[i]);
        Mat returnPoint = inv_param*tempMat;//역행렬을 곱해서 homogeneous좌표로 바꿉니다.
        homoOutputPoints.push_back(Point3d(returnPoint));
    }
    convertPointsFromHomogeneous(homoOutputPoints,arg_OuputPoints);
}

void Custom_triangulation(vector<Point2f> arg_InputPoints1,vector<Point2f> arg_InputPoints2, Mat arg_R, Mat arg_T,vector<Point3f> &arg_OuputPoints)
{
    int N = arg_InputPoints1.size();
    
    vector<Point3f> homoInputPoints1,homoInputPoints2;
    convertPointsToHomogeneous(arg_InputPoints1,homoInputPoints1);
    convertPointsToHomogeneous(arg_InputPoints2,homoInputPoints2);
    for(int i=0; i<N; i++)
    {
        //s_2* x_1^T * R* x_2 + x_1^* T를 최소화하는 s_2를 구한다.
        Mat x_1(homoInputPoints1[i]);
        Mat x_2(homoInputPoints2[i]);
        x_1.convertTo(x_1, CV_64F);
        x_2.convertTo(x_2, CV_64F);

        Mat s_2 = - (x_1.t()*arg_T)/(x_1.t()*arg_R*x_2);
        auto scalar_s_2 = s_2.at<float>(0,0);
        
    }
}

void CheckRT(int solution_idx,
            vector<Point2d> current_p2f, 
            vector<Point2d> past_p2f,
            Mat IntrinsicParam,
            Mat R, 
            Mat t, 
            vector<int> * good_point_ind,
            vector<Point3d> *current_good_point_3d)
{
  Mat projectMatrix = Mat::zeros(Size(4,3),CV_64FC1);
  R.copyTo(projectMatrix(Rect(0,0,3,3)));
  
  t.copyTo(projectMatrix(Rect(3,0,1,3)));
  
  Mat InitProjectMatrix = Mat::eye(3,4,CV_64F);
  Mat dist_coef = Mat::zeros(1,4,CV_64F);// 0값으로 초기화하였다.
  vector<Point2d> Undistorted_current_pt;
  vector<Point2d> Undistorted_past_pt;
  
    /* code */
    
  Custom_undisortionPoints(current_p2f,IntrinsicParam,Undistorted_current_pt);//mm단위로 바꿔줍니다.
  
  Custom_undisortionPoints(past_p2f,IntrinsicParam,Undistorted_past_pt);//mm단위로 바꿔줍니다.
  
  Mat InstrincParam_64FC1 =IntrinsicParam;
  vector<Point3f> Output_pt;
  Mat outputMatrix;
  triangulatePoints(InitProjectMatrix,projectMatrix,Undistorted_past_pt,Undistorted_current_pt,outputMatrix);
  vector<Point3d> points;
  
  for(int point_ind=0; point_ind<outputMatrix.cols; point_ind++)
  {
    Mat x = outputMatrix.col(point_ind);
    x /= x.at<double>(3,0);
    Point3d p (
          x.at<double>(0,0), 
          x.at<double>(1,0), 
          x.at<double>(2,0) 
      );
    points.push_back( p );
  }
  int good_match =0;
  //parallax 변수 : 각 카메라별 좌표 
  Mat PastPt = Mat::zeros(3,1, CV_64F);
  Mat CurrentPt = -R.t()*t;//과거 점 기준
  
  for(int point_ind=0; point_ind<points.size(); point_ind++)
  {
    Point3d point3d1 = points[point_ind];//3차원점입니다. 과거기준 좌표입니다. 
    Mat point_pose_in_past = Mat(point3d1);
    Mat estimated_point2 = R*(Mat(point3d1)+t);//현재 기준 좌표

    Mat PastNormal = point_pose_in_past - PastPt;
    Mat CurrentNormal = estimated_point2 - CurrentPt;
    float past_dist = norm(PastNormal);
    float current_dist = norm(CurrentNormal);

    float cosParallax = PastNormal.dot(CurrentNormal)/(past_dist*current_dist);
    if(cosParallax> 0.9999)
    {
      continue;
    }
    Point3d point3d2 (//현재기준 좌표입니다.
          estimated_point2.at<double>(0,0), 
          estimated_point2.at<double>(1,0), 
          estimated_point2.at<double>(2,0) 
      );
    double origin_depth = estimated_point2.at<double>(2,0);

    point3d1 /= point3d1.z; //정규화를 합니다.
    point3d2 /= point3d2.z; //정규화를 합니다. 
    Mat projected_pixel_point1 = InstrincParam_64FC1*Mat(point3d1);//카메라 파라미터를 곱해서 원래 과거 픽셀좌표로 바꿉니다. 
    Mat projected_pixel_point2 = InstrincParam_64FC1*Mat(point3d2);//카메라 파라미터를 곱해서 원래 현재 픽셀좌표로 바꿉니다. 

    double image1_error = (current_p2f[point_ind].x-projected_pixel_point2.at<double>(0,0))*
                                        (current_p2f[point_ind].x-projected_pixel_point2.at<double>(0,0))
                        + (current_p2f[point_ind].y-projected_pixel_point2.at<double>(1,0))*
                                        (current_p2f[point_ind].y-projected_pixel_point2.at<double>(1,0));

    double image2_error = (past_p2f[point_ind].x-projected_pixel_point1.at<double>(0,0))*
                                        (past_p2f[point_ind].x-projected_pixel_point1.at<double>(0,0))
                        + (past_p2f[point_ind].y-projected_pixel_point1.at<double>(1,0))*
                                        (past_p2f[point_ind].y-projected_pixel_point1.at<double>(1,0));
    if(origin_depth >0)//1. 3차원으로 z값이 0보다 작으면 잘못된 매칭입니다.
    {
      if(image1_error<3 && image2_error<3)
      {
        good_point_ind->push_back(point_ind);
        current_good_point_3d->push_back(point3d2);
      }
    }
  }

}

void CheckHomography(vector<Point2d> current_p2f, vector<Point2d> reference_p2f, double *Sh,Mat *HomographyMat,vector<int> *inliers) 
{
  float Th = 5.99;

  Mat H = findHomography(reference_p2f, current_p2f, RANSAC);
  H.convertTo(*HomographyMat,CV_64FC1);
  
  vector<Point3d> homo_current_p2f, homo_reference_p2f;
  convertPointsToHomogeneous(current_p2f,homo_current_p2f);
  convertPointsToHomogeneous(reference_p2f,homo_reference_p2f);
  double Pm_d2 =0;
  for(int i=0; i<reference_p2f.size(); i++)
  {
    bool is_outlier= true;
    //현재 프레임 좌표를 레퍼런스 프레임 좌표로 투영합니다. 
    Mat projected_currrent_p2f = *HomographyMat*Mat(homo_reference_p2f[i]);
    Point3d current_point3f (//현재기준 좌표입니다.
          projected_currrent_p2f.at<double>(0,0), 
          projected_currrent_p2f.at<double>(1,0), 
          projected_currrent_p2f.at<double>(2,0) 
      );
    current_point3f /=current_point3f.z;
    double diff = pow(norm(homo_current_p2f[i]-current_point3f),2);
    
    if(diff < Th)
    {
      Pm_d2 +=(Th-diff);
      is_outlier = false;
    }

    Mat projected_reference_p2f = HomographyMat->inv()*Mat(homo_current_p2f[i]);
    Point3d reference_point3f (//현재기준 좌표입니다.
          projected_reference_p2f.at<double>(0,0), 
          projected_reference_p2f.at<double>(1,0), 
          projected_reference_p2f.at<double>(2,0) 
      );
    reference_point3f /=reference_point3f.z;
    diff = pow(norm(homo_reference_p2f[i]-reference_point3f),2);
    if(diff < Th)
    {
      Pm_d2 +=(Th-diff);
    }
    else
    {
      is_outlier = true;
    }
    if(!is_outlier)
    {
      inliers->push_back(i);
    }
  }
  
  *Sh = Pm_d2;
}

void CheckFundamental(vector<Point2d> current_p2f, vector<Point2d> reference_p2f,Mat InstrinicParam,double *Sf,Mat *FundamentalMat,vector<int> *inliers) 
{
  float Th = 5.99;
  float Tf = 3.84;
  Mat EssentialMat = findEssentialMat(reference_p2f, current_p2f, InstrinicParam,RANSAC);
  EssentialMat.convertTo(EssentialMat,CV_64FC1);
  *FundamentalMat = (InstrinicParam.t()).inv()*EssentialMat*InstrinicParam.inv();
  
  vector<Point3d> homo_current_p2f, homo_reference_p2f;
  convertPointsToHomogeneous(current_p2f,homo_current_p2f);
  convertPointsToHomogeneous(reference_p2f,homo_reference_p2f);
  double Pm_d2 =0;
  for(int i=0; i<reference_p2f.size(); i++)
  {
    bool is_outlier= true;
    //현재 프레임 좌표를 레퍼런스 프레임 좌표로 투영합니다. 
    Mat projected_currrent_p2f = (*FundamentalMat)*Mat(homo_reference_p2f[i]);
    Point3d current_point3f (//현재기준 좌표입니다.
          projected_currrent_p2f.at<double>(0,0), 
          projected_currrent_p2f.at<double>(1,0), 
          projected_currrent_p2f.at<double>(2,0) 
      );
    //homo_current_p2f[i]-current_point3f
    double diff = current_point3f.x*(homo_current_p2f[i].x)+current_point3f.y*(homo_current_p2f[i].y)+current_point3f.z;
    diff = pow(diff,2)/(current_point3f.x*current_point3f.x+current_point3f.y*current_point3f.y);
    if(diff < Tf)
    {
      Pm_d2 +=(Th-diff);
      is_outlier = false;
    }
    Mat projected_reference_p2f = FundamentalMat->t()*Mat(homo_current_p2f[i]);
    Point3f reference_point3f (//현재기준 좌표입니다.
          projected_reference_p2f.at<double>(0,0), 
          projected_reference_p2f.at<double>(1,0), 
          projected_reference_p2f.at<double>(2,0) 
      );
    diff = reference_point3f.x*(homo_reference_p2f[i].x)+reference_point3f.y*(homo_reference_p2f[i].y)+reference_point3f.z;
    diff = pow(diff,2)/(reference_point3f.x*reference_point3f.x+reference_point3f.y*reference_point3f.y);
    if(diff < Tf)
    {
      Pm_d2 +=(Th-diff);
    }
    else
    {
      is_outlier = true;
    }
    if(!is_outlier)
    {
      inliers->push_back(i);
    }
  }
  
  *Sf = Pm_d2;
}

bool ValidateHomographyRt(vector<Point2d> &arg_kp1, //현재 카메라의 키포인트들의 좌표입니다. arg_kp1과 arg_kp2는 서로 순서대로 매칭되어있습니다.
                          vector<Point2d> &arg_kp2, //과거 카메라의 키포인트들의 좌표입니다.arg_kp1과 arg_kp2는 서로 순서대로 매칭되어있습니다.
                          Mat InstrincParam, //카메라의 내개파라미터입니다.(카메라 행렬)
                          Mat& R, //현재 Homography에 대한 정답 R행렬을 반환합니다.
                          Mat& t, //현재 Homography에 대한 정답 T행렬을 반환합니다.
                          vector<int> &good_point_ind, //현재 Homography에 대해서 좋은 매칭을 가지는 점의 인덱스를 반환합니다. 인덱스는 arg_kp1과 arg_kp2내에서의 인덱스를 의미합니다.
                          vector<Point3d> &current_good_point_3d) //현재 Homography에 대해서 좋은 매칭을 가지는 점의 3차원점을 반환합니다. 3차원점은 현재 카메라 좌표계를 기준으로 반환됩니다. 
{
  Mat H = findHomography(arg_kp2, arg_kp1, RANSAC);//현재 Homography입니다. Opencv의 내장함수를 통해서 빠르게 구하며, RANSAC을 사용하여 구합니다.
  vector<Mat> Rs_decomp; //Homography를 decompose하여 R행렬의 4가지의 해 후보를 얻습니다. 
  vector<Mat> Ts_decomp; //Homography를 decompose하여 T행렬의 4가지의 해 후보를 얻습니다. 
  vector<Mat> Normals_decomp; //Homography를 decompose하여 N행렬의 4가지의 해 후보를 얻습니다. 
  int solutions = decomposeHomographyMat(H,InstrincParam,Rs_decomp,Ts_decomp,Normals_decomp);//Homography matrix를 decompose하여 4가지해를 얻습니다. 반환값은 해의개수(=4)입니다.
  vector<vector<int>> good_points_ind(solutions);//0번해에 대해서 좋은매칭을 가지는 점의 인덱스 배열(벡터)입니다. arg_kp1에서의 인덱스입니다.
  vector<vector<Point3d>> current_good_points_3d(solutions);
  thread threads[solutions];//4가지의 thread로 각 해에 대한 답을 얻습니다. 
  for(int i=0; i<solutions; i++)
  {
    threads[i] =  thread(CheckRT,i,arg_kp1,arg_kp2,InstrincParam,Rs_decomp[i],Ts_decomp[i], &good_points_ind[i],&current_good_points_3d[i]); 
  }
  
  int* good_points = new int(solutions);
  for(int i =0; i<solutions; i++)
  {
    // good_points
    threads[i].join();
  }
  for(int i=0; i<4; i++)
  {
    
  }
  int max_ind =-1;
  int good_point_size =-1;
  //가장 매칭점을 많이 갖는 해를 정답인 해로 결정합니다.
  for(int j=0; j<solutions; j++)
  {
    if(good_points_ind[j].size() - good_point_size>0)
    {
      max_ind =j;
      good_point_size= good_points_ind[j].size();
    }
  }

  //만약에 없다면, 코드를 종료합니다. 
  if(max_ind == -1)
  {
    cout<<" 호모그래피에서 매칭되는 점이 하나도 없습니다. 종료"<<endl;
    exit(0);
  }
  else
  {//만약에 있다면, R,t, 매칭점인덱스, 3차원 매칭점을 반환합니다.  
    R = Rs_decomp[max_ind];
    t = Ts_decomp[max_ind];
    good_point_ind = good_points_ind[max_ind];
    current_good_point_3d = current_good_points_3d[max_ind];
  }
  
}

bool ValidateFundamentalRt(vector<Point2d> &arg_kp1, 
                        vector<Point2d> &arg_kp2, 
                        Mat InstrincParam,
                        Mat& R, 
                        Mat& t,
                        vector<int> &good_point_ind, //현재 Fundamental에 대해서 좋은 매칭을 가지는 점의 인덱스를 반환합니다. 인덱스는 arg_kp1과 arg_kp2내에서의 인덱스를 의미합니다.
                        vector<Point3d> &current_good_point_3d) //현재 Fundamental에 대해서 좋은 매칭을 가지는 점의 3차원점을 반환합니다. 3차원점은 현재 카메라 좌표계를 기준으로 반환됩니다.  
{
  Mat opencv_essential_mat;
  opencv_essential_mat=findEssentialMat(arg_kp2,arg_kp1,InstrincParam);
  Mat outputR;
  Mat outputT;
  Mat Mask;
  // recoverPose(opencv_essential_mat,arg_kp2,arg_kp1,InstrincParam,outputR,outputT,Mask);
  recoverPose(opencv_essential_mat,arg_kp2,arg_kp1,InstrincParam,outputR,outputT);
  thread threads;
  
  threads =  thread(CheckRT,4,arg_kp1,arg_kp2,InstrincParam,outputR,outputT, &good_point_ind,&current_good_point_3d); 
  threads.join();
  
  R = outputR;
  t = outputT;
  
}

void ExtractPoint2D(char* filename, vector<Point2d> &extracted_point2d, int &image_width, int &image_height, float &arg_min_point_x, float& arg_min_point_y)
{
  extracted_point2d.clear();//기존에 들어있는 점을 지웁니다.
  ifstream poseFile;
  poseFile.open(filename);
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
  image_width = (int)((max_point_x- min_point_x)*3+20);//좌우로 10칸씩 추가하였다. 
  image_height = (int)(max_point_y - min_point_y)*30+20;//좌우로 10칸씩 추가하였다. 행은 10배로 늘렸다.
  arg_min_point_x = min_point_x;
  arg_min_point_y = min_point_y;
}

void ExtractPoint3D(char* filename, 
                    vector<Point3f> &extracted_point3d, 
                    int &image_width, 
                    int &image_height, 
                    float &arg_min_point_x, 
                    float &arg_min_point_z) 
{
  extracted_point3d.clear();//기존에 들어있는 점을 지웁니다.
  ifstream poseFile;
  poseFile.open(filename);
  if(poseFile.is_open())
  {
    while(!poseFile.eof())
    {
      string mat_pose;
      float p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12;
      
      getline(poseFile,mat_pose);
      sscanf(mat_pose.c_str(),"%e %e %e %e %e %e %e %e %e %e %e %e",&p1,&p2,&p3,&p4,&p5,&p6,&p7,&p8,&p9,&p10,&p11,&p12);
      Point3f sample_point = Point3f(p4,p8,p12);//x,y좌표이다.
      extracted_point3d.push_back(sample_point);
    }
  }
  else
  {
    cout<<"외장하드가 연결이 안된듯합니다."<<endl;
    exit(0);
  }

  float max_point_x = -10000;
  float min_point_x = 10000;
  float max_point_z = -10000;
  float min_point_z = 10000;
  for(auto point = extracted_point3d.begin(); point != extracted_point3d.end(); point++)
  {
      if(point->x > max_point_x)
      {
        max_point_x = point->x;
      }
      if(point->z > max_point_z)
      {
        max_point_z = point->z;
      }
      if(point->x < min_point_x)
      {
        min_point_x = point->x;
      }
      if(point->z < min_point_z)
      {
        min_point_z = point->z;
      }
  }
  image_width = (int)((max_point_x- min_point_x)*1+20);//좌우로 10칸씩 추가하였다. 
  image_height = (int)(max_point_z - min_point_z)*1+20;//좌우로 10칸씩 추가하였다. 행은 10배로 늘렸다.
  arg_min_point_x = min_point_x;
  arg_min_point_z = min_point_z;
}