#pragma once
#include "KeyFrame.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mutex>

using namespace std;
using namespace cv;
class Match_Set
{
  //같은 지점을 본 키포인트 들입니다.
  public:
    Match_Set(KeyFrame* arg_FirstFrame,
              KeyFrame* arg_SecondFrame,
              KeyPoint* arg_FirstPoint, 
              KeyPoint* arg_SecondPoint)
              :kf_first(arg_FirstFrame),
              kf_second(arg_SecondFrame),
              kp_first(arg_FirstPoint),
              kp_second(arg_SecondPoint){}
    KeyFrame* kf_first;
    KeyFrame* kf_second;
    //서로가 본 키포인트 위치입니다. 
    KeyPoint* kp_first;
    KeyPoint* kp_second;
};
class NewKeyFrameSet
{
  public:
    KeyFrame* CurrentFrame; //래퍼런스와의 매칭간에 현재 프레임의 포인터입니다.
    KeyFrame* ReferenceFrame;//래퍼런스와의 매칭간에 래퍼런스 프레임의 포인터입니다.
    vector<Point2f> CurrentGoodPoint2D;//래퍼런스와의 매칭간에 좋은 매칭을 가지는 현재 2d점입니다.
    vector<Point2f> ReferenceGoodPoint2D;//래퍼런스와의 매칭간에 좋은 매칭을 가지는 레퍼런스 2d점입니다.

    vector<Point3d> CurrentGoodPoint3D;//래퍼런스와의 매칭간에 좋은 매칭을 가지는 현재 3d점입니다.
    vector<Mat> descriptor;
    Mat R;//래퍼런스와의 매칭간에 좋은 매칭을 가지는 R입니다.
    Mat t;//래퍼런스와의 매칭간에 좋은 매칭을 가지는 t입니다.
};

class NodeHandler//맵포인트를 전반적으로 관리, 평가합니다. 
{
  public://variable
    //비교하는 부분을 윈도우를 만들어서 그걸로 비교하자.
    vector<MapPoint*> _pt_MapPoints;//전체 맵포인트가 들어있습니다.
    vector<KeyFrame*> _pt_KeyFrames;//전체 키프레임이 담겨있습니다. 
    vector<KeyFrame*> _pt_LocalWindowKeyFrames;//윈도우 단위로 키 프레임이 담겨있습니다. 키프레임이 일정이상 생기면 일정단위로 삭제합니다. 
    vector<MapPoint*> _pt_LocalWindowMapPoints;//윈도우 단위로 MapPoint를 유지합니다. 키프레임이 일정이상 생기면 일정단위로 삭제합니다. 
    vector<NewKeyFrameSet*> v_newKeyFrame;
    Ptr<DescriptorMatcher> _match_OrbMatchHandle = BFMatcher::create(NORM_HAMMING);
    Mat _mat_FullDescriptor;//전체 descriptor입니다. 
    Mat _mat_LocalDescriptor; //윈도우 단위로 descriptor를 유지합니다.
    
    int int_CurrentFrameIdx = 0;
    int int_LastKeyFrameIdx = -1;

    bool _b_IsSetInstricParam= false;
    Mat mat_InstrisicParam;
    int _int_MapPointIdx = 0;
    int _int_FrameThreshold = 20;
    int _int_LocalWindowSize =20;
    vector<Match_Set*> _local_MatchSet;
    mutex m_sharedlock;
    vector<Mat> estimated_pose;
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
      mat_InstrisicParam =(Mat_<float>(3,3) <<arg_f_x,arg_skef_cf_x,arg_c_x,
                                              0,arg_f_y,arg_c_y,
                                              0,0,1);
      _b_IsSetInstricParam =true;
      }
    bool Make_MapPoint_pix2pixMatch(Mat arg_descriptor,vector<KeyPoint> arg_KeyPoint, KeyFrame* arg_KeyFrame);

    bool ValidateAndAddFrame(Mat arg_candidateImage);
    bool MakeMapPoint(KeyFrame* kfp_beforeFrame, KeyFrame* kfp_afterFrame);

    vector<KeyFrame*> Get_LocalKeyFrame(void);
    bool Vector2Mat_p2fMat(vector<Point2f> arg_vectorpt2, Mat arg_Mat);
    bool IsNiceTime();
    bool GetLastFrame(KeyFrame* &p_lastFrame);
    bool AddNewKeyFrame(KeyFrame* p_NewFrame);
    bool SetImageFeature(KeyFrame* p_NewFrame, Mat Image);
    void SetRt(KeyFrame* p_TempFrame, Mat R,Mat T);
    void GetRtParam(KeyFrame* p_TempFrame, float* &R_tparam);
    vector<MapPoint*> Get_localMapPoint();
    void Add_CandidateKeyFrame(NewKeyFrameSet* matches);
    //Todo : 후보키프레임 등록하는부분을 해야함. 기존에 단순히 add하는거랑 다름
};