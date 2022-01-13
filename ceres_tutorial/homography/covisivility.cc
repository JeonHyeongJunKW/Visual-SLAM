#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

// ORB를 사용하여 feature 추출
void extract(const Mat& img, Mat&des, vector<KeyPoint>& kp)
{
  const static auto& orb = ORB::create();
  orb->detectAndCompute(img,noArray(),kp,des); // no mask
}

int main(int argc, char** argv )
{
  // termenal에서 실행파일 이름이 argv[0]를 점유함
    Mat frames[11] = {
      imread("../000100.png",IMREAD_GRAYSCALE), 
      imread("../000101.png",IMREAD_GRAYSCALE), 
      imread("../000102.png",IMREAD_GRAYSCALE), 
      imread("../000103.png",IMREAD_GRAYSCALE), 
      imread("../000104.png",IMREAD_GRAYSCALE), 
      imread("../000105.png",IMREAD_GRAYSCALE), 
      imread("../000106.png",IMREAD_GRAYSCALE), 
      imread("../000107.png",IMREAD_GRAYSCALE), 
      imread("../000108.png",IMREAD_GRAYSCALE), 
      imread("../000109.png",IMREAD_GRAYSCALE), 
      imread("../000110.png",IMREAD_GRAYSCALE)
    };
    Mat descriptors[11], Results[11], perspective[11];
    std::vector<vector<KeyPoint>> kps(11);
    Ptr<DescriptorMatcher> Match_ORB = BFMatcher::create(NORM_HAMMING2);

    extract(frames[0], descriptors[0], kps[0]);
    for(int i = 1; i < 11; i++){
      std::vector<vector<DMatch>> matches;
      extract(frames[i], descriptors[i], kps[i]);

      // 배열 간의 Hamming 거리를 계산합니다.
      Match_ORB->knnMatch(descriptors[0], descriptors[i], matches, 2); // no mask
      const int match_size = matches.size();    // 이 인자가 의미하는 바를 잘 모르겠음
      sort(matches.begin(),matches.end());

      // 
      Mat Result;
      vector<DMatch> good_matches;
      vector<KeyPoint> kps1, kps2;
      const float ratio_thresh = 0.75f;
      for (size_t j = 0; j < matches.size(); j++)
      {
          if (matches[j][0].distance < ratio_thresh * matches[j][1].distance) // m.d[1]/m.d[1] < 0.75
          {
              good_matches.push_back(matches[j][0]);
              KeyPoint kp1 = kps[0].at(matches[j][0].queryIdx);
              kps1.push_back(kp1);
              KeyPoint kp2 = kps[i].at(matches[j][0].trainIdx);
              kps2.push_back(kp2);
              // cout << kp1.pt << " / " << kp2.pt << endl;
          }
      }

      cv::drawMatches(frames[0], kps[0], frames[i], kps[i], good_matches, Result, cv::Scalar::all(-1),cv::Scalar(-1), vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      // cv::imshow("result", Result);

      cv::drawKeypoints(frames[0], kps1, Results[0]);
      cv::drawKeypoints(frames[i], kps2, Results[i]);

      vector<Point2f> first_point, second_point;
      for( size_t j = 0; j < good_matches.size(); j++ )
      {
          //-- Get the keypoints from the good matches
          first_point.push_back( kps[0][ good_matches[j].queryIdx ].pt );
          second_point.push_back( kps[i][ good_matches[j].trainIdx ].pt );
          // cout << first_point.at(i)<< " / " << second_point.at(i) << endl;
      }
      Mat H = findHomography( first_point, second_point, RANSAC);
      cout<< "Homography" <<endl;
      cout<< H << endl;

      warpPerspective(Results[0], perspective[i], H, Results[0].size());
    }

    for(int i = 1; i < 11; i++){
      imwrite("result" + to_string(i) + ".jpg", Results[i]);
      imwrite("perspective" + to_string(i) + ".jpg", perspective[i]);
    } 

    cv::drawKeypoints(frames[0], kps[0], Results[0]);
    imwrite("result" + to_string(0) + ".jpg", Results[0]);
    // while(waitKey(1) == -1)
    // {
    //   continue;
    // }
    return 0;
}

/*int main(){
  cv:Mat_<double> Essential;
  Essential <<  0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1;
  Mat R1, R2, t;
  cv::decomposeEssentialMat(Essential, R1, R2, t);
  cout << "R1 \n" << R1 << "\n";
  cout << "R2 \n" << R2 << "\n";
  cout << "t \n" << t << "\n";
}*/