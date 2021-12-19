#pragma once
#include "KeyFrame.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class NodeHandler//맵포인트를 전반적으로 관리, 평가합니다. 
{
  private://variable
    vector<MapPoint*> _pt_MapPoints;
    vector<KeyFrame*> _pt_KeyFrames;
    
    Ptr<DescriptorMatcher> _match_OrbMatchHandle = BFMatcher::create(NORM_HAMMING);
    Mat _full_descirptor;//전체 descriptor입니다. 
    int int_CurrentFrameIdx = 0;
    int int_LastKeyFrameIdx = -1;

  // public://constructor 
  //   NodeHander();
  private: 
    bool _Make_NewMapPoint(Mat arg_Descriptor, MapPoint* &arg_OutputMapPoint);
    //새로운 맵포인트를 만듭니다.
    bool _Is_NewMapPoint(Mat arg_Descriptor, MapPoint* &arg_Old_MapPoint);
    void _Apply_DescriptorMat(Mat arg_Descriptor);
    int _Get_NumberOfOrbFeature(Mat arg_candidateImage, Mat&des, vector<KeyPoint>& kp);
    bool _Is_TrackingMapPoint(Mat arg_Descriptor);

  public://method
    bool Make_MapPoint(Mat arg_Descriptor, MapPoint* &arg_OutputMapPoint);
    //맵포인트를 추가합니다. 만약에 이미 유사한 맵포인트가 있다면 해당 맵포인트의 pointer를 반환합니다.
    
    bool Add_MapPoint();//
    bool Delete_MapPoint(MapPoint* arg_ptMapPoint);// 맵포인트를 삭제합니다.
    bool Is_GoodKeyFrame(Mat arg_candidateImage);
    int Get_MapPointSize(void); // 맵포인트의 사이즈를 반환합니다.
};