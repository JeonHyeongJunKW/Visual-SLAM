#include "CameraTool.h"
#include <iostream>
#include <opencv2/opencv.hpp>

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

void Custom_undisortionPoints(vector<Point2f> arg_InputPoints, Mat IntrinsicParam,vector<Point2f> &arg_OuputPoints)
{
    int N = arg_InputPoints.size();
    vector<Point3f> homoInputPoints;
    convertPointsToHomogeneous(arg_InputPoints,homoInputPoints);
    Mat inv_param = IntrinsicParam.inv();
    vector<Point3f> homoOutputPoints;
    for(int i=0; i< N; i++)
    {
        Mat tempMat(homoInputPoints[i]);
        // cout<<tempMat<<endl;
        Mat returnPoint = inv_param*tempMat;//역행렬을 곱해서 homogeneous좌표로 바꿉니다.
        homoOutputPoints.push_back(Point3f(returnPoint));
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
        // cout<<x_1.cols<<endl;
        // cout<<x_2.cols<<endl;
        // cout<<arg_T.cols<<endl;
        // cout<<arg_R.cols<<endl;
        Mat s_2 = - (x_1.t()*arg_T)/(x_1.t()*arg_R*x_2);
        auto scalar_s_2 = s_2.at<float>(0,0);
        // cout<<"-------------"<<endl;
        // cout <<"scalar : "<<scalar_s_2<<endl;
        // if(i<5)
        // {
        //     cout<<"real 3d : "<<scalar_s_2*x_2.t()<<endl;
        // }
        
    }
}