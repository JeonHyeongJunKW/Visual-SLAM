#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
void ExtractPoint2D(char* filename, 
                    vector<Point2f> &extracted_point2d, 
                    int &image_width, 
                    int &image_height, 
                    float &arg_min_point_x, 
                    float& arg_min_point_y);
void ExtractPoint3D(char* filename, 
                    vector<Point3f> &extracted_point3d, 
                    int &image_width, 
                    int &image_height, 
                    float &arg_min_point_x, 
                    float& arg_min_point_z);

void Custom_undisortionPoints(vector<Point2d> arg_InputPoints, Mat IntrinsicParam,vector<Point2d> &arg_OuputPoints);
void Custom_triangulation(vector<Point2f> arg_InputPoints1,vector<Point2f> arg_InputPoints2, Mat arg_R, Mat arg_T,vector<Point3f> &arg_OuputPoints);
string type2str(int type);
void CheckRT(int solution_idx,
            vector<Point2d> current_p2f, 
            vector<Point2d> past_p2f,
            Mat IntrinsicParam,
            Mat R, 
            Mat t, 
            vector<int> * good_point_ind, 
            vector<Point3d> *current_good_point_3d);
void CheckHomography(vector<Point2d> current_p2f, 
                    vector<Point2d> reference_p2f,
                    double *Sh,
                    Mat *HomographyMat,
                    vector<int> *inliers);
                    
void CheckFundamental(vector<Point2d> current_p2f, 
                    vector<Point2d> reference_p2f,
                    Mat InstrinicParam,double *Sf,
                    Mat *FundamentalMat,
                    vector<int> *inliers);

bool ValidateHomographyRt(vector<Point2d> &arg_kp1, //현재 카메라의 키포인트들의 좌표입니다. arg_kp1과 arg_kp2는 서로 순서대로 매칭되어있습니다.
                        vector<Point2d> &arg_kp2, //과거 카메라의 키포인트들의 좌표입니다.arg_kp1과 arg_kp2는 서로 순서대로 매칭되어있습니다.
                        Mat InstrincParam,//카메라의 내부파라미터입니다.(카메라 행렬)
                        Mat& R, //현재 Homography에 대한 정답 R행렬을 반환합니다.
                        Mat& t,//현재 Homography에 대한 정답 T행렬을 반환합니다.
                        vector<int> &good_point_ind, //현재 Homography에 대해서 좋은 매칭을 가지는 점의 인덱스를 반환합니다. 인덱스는 arg_kp1과 arg_kp2내에서의 인덱스를 의미합니다.
                        vector<Point3d> &current_good_point_3d); //현재 Homography에 대해서 좋은 매칭을 가지는 점의 3차원점을 반환합니다. 3차원점은 현재 카메라 좌표계를 기준으로 반환됩니다. 
bool ValidateFundamentalRt(vector<Point2d> &arg_kp1, 
                        vector<Point2d> &arg_kp2, 
                        Mat InstrincParam,
                        Mat& R, 
                        Mat& t,
                        vector<int> &good_point_ind, //현재 Fundamental에 대해서 좋은 매칭을 가지는 점의 인덱스를 반환합니다. 인덱스는 arg_kp1과 arg_kp2내에서의 인덱스를 의미합니다.
                        vector<Point3d> &current_good_point_3d); //현재 Fundamental에 대해서 좋은 매칭을 가지는 점의 3차원점을 반환합니다. 3차원점은 현재 카메라 좌표계를 기준으로 반환됩니다. 
