#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Custom_undisortionPoints(vector<Point2f> arg_InputPoints, Mat IntrinsicParam,vector<Point2f> &arg_OuputPoints);
void Custom_triangulation(vector<Point2f> arg_InputPoints1,vector<Point2f> arg_InputPoints2, Mat arg_R, Mat arg_T,vector<Point3f> &arg_OuputPoints);
string type2str(int type);