#include "CameraTool.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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

void Custom_triangulation(vector<Point2f> arg_InputPoints1,vector<Point2f> arg_InputPoints2, Mat ProjectionMatrix,vector<Point3f> &arg_OuputPoints)
{

}