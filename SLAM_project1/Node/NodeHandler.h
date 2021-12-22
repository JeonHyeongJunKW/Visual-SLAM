#pragma once
#include "KeyFrame.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class NodeHandler//맵포인트를 전반적으로 관리, 평가합니다. 
{
  private://variable
    //TODO
    //오늘 수정할 부분
    //비교하는 부분을 윈도우를 만들어서 그걸로 비교하자.
    vector<MapPoint*> _pt_MapPoints;//전체 맵포인트가 들어있습니다.
    vector<KeyFrame*> _pt_KeyFrames;//전체 키프레임이 담겨있습니다. 
    vector<KeyFrame*> _pt_LocalWindowKeyFrames;//윈도우 단위로 키 프레임이 담겨있습니다. 키프레임이 일정이상 생기면 일정단위로 삭제합니다. 
    vector<MapPoint*> _pt_LocalWindowMapPoints;//윈도우 단위로 MapPoint를 유지합니다. 키프레임이 일정이상 생기면 일정단위로 삭제합니다. 
    
    Ptr<DescriptorMatcher> _match_OrbMatchHandle = BFMatcher::create(NORM_HAMMING);
    Mat _mat_FullDescriptor;//전체 descriptor입니다. 
    Mat _mat_LocalDescriptor; //윈도우 단위로 descriptor를 유지합니다.
    
    int int_CurrentFrameIdx = 0;
    int int_LastKeyFrameIdx = -1;

    bool _b_IsSetInstricParam= false;
    Mat _mat_InstrisicParam;
    int _int_MapPointIdx = 0;
    int _int_FrameThreshold = 20;
    int _int_LocalWindowSize =20;

  public://constructor 
    NodeHandler();
    NodeHandler(int arg_FrameThreshold);
    NodeHandler(int arg_FrameThreshold, int arg_LocalWindowSize);
  private: 
    bool _Make_NewMapPoint(Mat arg_Descriptor, MapPoint* &arg_OutputMapPoint);
    //새로운 맵포인트를 만듭니다.
    bool _Is_NewMapPoint(Mat arg_Descriptor, MapPoint* &arg_Old_MapPoint);
    void _Apply_DescriptorMat(Mat arg_Descriptor);
    int _Get_NumberOfOrbFeature(Mat arg_candidateImage, Mat&des, vector<KeyPoint>& kp);
    bool _Is_TrackingMapPoint(Mat arg_Descriptor);

  public://method
    void Set_InstricParam(float arg_f_x, float arg_f_y, float arg_skef_cf_x, float arg_c_x, float arg_c_y) {
      _mat_InstrisicParam =(Mat_<float>(3,3) <<arg_f_x,arg_skef_cf_x,arg_c_x,0,arg_f_y,arg_c_y,0,0,1);
      _b_IsSetInstricParam =true;
      }
    bool Make_KeyFrame(Mat arg_KeyFrame);
    bool Make_MapPoint(Mat arg_Descriptor, MapPoint* &arg_OutputMapPoint);
    //맵포인트를 추가합니다. 만약에 이미 유사한 맵포인트가 있다면 해당 맵포인트의 pointer를 반환합니다.
    bool Add_MapPoint();//
    bool Delete_MapPoint(MapPoint* arg_ptMapPoint);// 맵포인트를 삭제합니다.
    bool Is_GoodKeyFrame(Mat arg_candidateImage);
    int Get_MapPointSize(void); // 맵포인트의 사이즈를 반환합니다.
    bool Make_MapPoint_pix2pixMatch(Mat arg_descriptor,vector<KeyPoint> arg_KeyPoint, KeyFrame arg_KeyFrame);
    void Change_Window(KeyFrame* arg_NewKeyFrame);
    void Delete_Descriptor(int arg_delIdx);
};