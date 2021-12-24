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

void KeyFrame::Add_MapPoint(MapPoint* arg_MapPoint)
{
  this->_pmappoint_OwnedMapPoint.push_back(arg_MapPoint);
}

void KeyFrame::Set_Descriptor(Mat arg_descriptors)
{
  this->_mat_descriptors = arg_descriptors;
}

void KeyFrame::Set_KeyPoint(vector<KeyPoint> arg_keyPoint)
{
  this->_vkey_keypoints = arg_keyPoint;
}

Mat KeyFrame::Get_Descriptor(void)
{
  return this->_mat_descriptors;
}
vector<KeyPoint> KeyFrame::Get_keyPoint(void)
{
  return this->_vkey_keypoints;
}