#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
void ExtractPoint2D(char* filename, vector<Point2f> &extracted_point2d, int &image_width, int &image_height, float &arg_min_point_x, float& arg_min_point_y);
void Custom_undisortionPoints(vector<Point2f> arg_InputPoints, Mat IntrinsicParam,vector<Point2f> &arg_OuputPoints);
void Custom_triangulation(vector<Point2f> arg_InputPoints1,vector<Point2f> arg_InputPoints2, Mat arg_R, Mat arg_T,vector<Point3f> &arg_OuputPoints);
string type2str(int type);
void CheckRT(int solution_idx,
            vector<Point2f> current_p2f, 
            vector<Point2f> past_p2f,
            Mat IntrinsicParam,
            Mat R, 
            Mat t, 
            vector<int> * good_point_ind, 
            vector<Point3d> *current_good_point_3d);
void CheckHomography(vector<Point2f> current_p2f, 
                    vector<Point2f> reference_p2f,
                    float *Sh,
                    Mat *HomographyMat,
                    vector<int> *inliers);
void CheckFundamental(vector<Point2f> current_p2f, 
                    vector<Point2f> reference_p2f,
                    Mat InstrinicParam,float *Sf,
                    Mat *FundamentalMat,
                    vector<int> *inliers);

bool ValidateHomographyRt(vector<Point2f> &arg_kp1, 
                        vector<Point2f> &arg_kp2, 
                        Mat InstrincParam,
                        Mat& R, 
                        Mat& t);
bool ValidateFundamentalRt(vector<Point2f> &arg_kp1, 
                        vector<Point2f> &arg_kp2, 
                        Mat InstrincParam,
                        Mat& R, 
                        Mat& t);
