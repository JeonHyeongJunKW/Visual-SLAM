#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

void extract(const Mat& img, Mat&des, vector<KeyPoint>& kp)
{
  const static auto& orb = ORB::create();
  orb->detectAndCompute(img,noArray(),kp,des);
}

int main(int argc, char** argv )
{
    Mat first_image = imread(argv[1],IMREAD_GRAYSCALE);
    Mat second_image = imread(argv[2],IMREAD_GRAYSCALE);
    // cout<<argv[0]<<" "<< argv[1]<<endl;
    Mat des_first, des_second;
    vector<KeyPoint> kp_first, kp_second;

    std::vector< vector<DMatch>> matches;

    extract(first_image,des_first,kp_first);
    extract(second_image,des_second,kp_second);
    Ptr<DescriptorMatcher> Match_ORB = BFMatcher::create(NORM_HAMMING);

    Match_ORB->knnMatch(des_first,des_second,matches,2);
    const int match_size = matches.size();
    sort(matches.begin(),matches.end());

    vector<DMatch> good_matches;
    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }
    

    cv::Mat Result;

    cv::drawMatches(first_image,kp_first,second_image,kp_second,good_matches,Result,cv::Scalar::all(-1),cv::Scalar(-1), vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("result",Result);
    vector<Point2f> first_point;
    vector<Point2f> second_point;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        first_point.push_back( kp_first[ good_matches[i].queryIdx ].pt );
        second_point.push_back( kp_second[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( first_point, second_point, RANSAC);
    cout<< "Homography" <<endl;
    cout<< H<<endl;
    while(waitKey(1) != 37)
    {
      continue;
    }
    return 0;
}