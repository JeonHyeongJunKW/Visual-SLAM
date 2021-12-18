#include <iostream>
#include <opencv2/opencv.hpp>
#include "KeyFrame.h"

using namespace std;
using namespace cv;


KeyFrame::KeyFrame(int arg_Key)
{
  this->_int_Keyindex=arg_Key;
}

KeyFrame::KeyFrame(int arg_Key, Mat arg_IntrinsicParam)
{
  this->_int_Keyindex=arg_Key;
  this->_mat_Intrinsicparam= arg_IntrinsicParam.clone();
}
KeyFrame::KeyFrame(int arg_Key, Mat arg_IntrinsicParam, Mat arg_OriginImage)
{
  this->_int_Keyindex=arg_Key;
  this->_mat_Intrinsicparam= arg_IntrinsicParam.clone();
  this->_mat_Originimage = arg_OriginImage.clone();
}
KeyFrame::KeyFrame(int arg_Key, Mat arg_IntrinsicParam, Mat arg_OriginImage,float* arg_camera_R_t)
{
  this->_int_Keyindex=arg_Key;
  this->_mat_Intrinsicparam= arg_IntrinsicParam.clone();
  this->_mat_Originimage = arg_OriginImage.clone();
  this->_pf_camera_R_t = arg_camera_R_t;
}

void KeyFrame::Set_IntrinsicParam(Mat arg_IntrinsicParam)
{
  this->_mat_Intrinsicparam= arg_IntrinsicParam.clone();
}

Mat KeyFrame::Get_IntrinsicParam(void)
{
  return this->_mat_Intrinsicparam;
}

void KeyFrame::Set_OriginImage(Mat arg_OriginImage)
{
  this->_mat_Originimage = arg_OriginImage.clone();
}

Mat KeyFrame::Get_OriginImage(void)
{
  return this->_mat_Originimage;
}

int KeyFrame::Get_KeyIndex(void)
{
  return this->_int_Keyindex;
}

KeyFrame* KeyFrame::Get_FatherKeyFrame(void)
{
  return this->_pkeyframe_Fatherkeyframe;
}
void KeyFrame::Set_FatherKeyFrame(KeyFrame* arg_FatherFrame)
{
  this->_pkeyframe_Fatherkeyframe = arg_FatherFrame;
}
KeyFrame* KeyFrame::Get_ChildKeyFrame(void)
{
  return this->_pkeyframe_Childkeyframe;
}
void KeyFrame::Set_ChildKeyFrame(KeyFrame* arg_ChildFrame)
{
  this->_pkeyframe_Childkeyframe = arg_ChildFrame;
}

vector<MapPoint*> KeyFrame::Get_MapPoint()
{
  return this->_pmappoint_OwnedMapPoint;
}

int KeyFrame::Get_NumMapPoint()
{
  return this->_pmappoint_OwnedMapPoint.size();
}

void KeyFrame::Make_MapPoint()//새로 맵포인트를 만듭니다.
{
  this->_pmappoint_OwnedMapPoint.clear();
  //1. 현재 이미지에서 feature를 뽑습니다. 
  //2. 그중에서 적당한 임계값을 가지는 OrbFeature를 뽑습니다. 
  //3. 이것을 이전에 노드들과 비교합니다. 
  //4. 이전 노드와 가진게 비슷한지도 비교합니다. 
}

void KeyFrame::Make_MapPoint(Mat arg_OriginImage)//새로 인자로 들어온 이미지를 가지고 맵포인트를 만듭니다.
{
  this->_mat_Originimage = arg_OriginImage.clone();
  this->_pmappoint_OwnedMapPoint.clear();
}