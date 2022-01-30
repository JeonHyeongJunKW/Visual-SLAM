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
KeyFrame::KeyFrame(int arg_Key, Mat arg_IntrinsicParam, Mat arg_OriginImage,double* arg_camera_R_t)
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

map<int, MapPoint*> KeyFrame::Get_MapPoint()
{
  return this->_pmappoint_OwnedMapPoint;
}

int KeyFrame::Get_NumMapPoint()
{
  return this->_pmappoint_OwnedMapPoint.size();
}

void KeyFrame::Add_MapPoint(MapPoint* arg_MapPoint)
{
  if(this->_pmappoint_OwnedMapPoint.find(arg_MapPoint->int_Node) != this->_pmappoint_OwnedMapPoint.end())
  {
    cout<<"already have error4"<<endl;
    cout<<arg_MapPoint->int_Node<<endl;
    cout<<this->Get_KeyIndex()<<endl;
    cout<<this->_pmappoint_OwnedMapPoint[arg_MapPoint->int_Node]->int_Node<<endl;
    exit(0);
  }
  else
  this->_pmappoint_OwnedMapPoint[arg_MapPoint->int_Node] = arg_MapPoint;//노드 이름으로 저장됩니다. 
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
void KeyFrame::Set_Rt(Mat arg_R, Mat arg_T)
{
  //포인터 할당합니다.
  this->_pf_camera_R_t = new double[13];
  // arg_R.convertTo(arg_R, CV_32FC1);
  // arg_T.convertTo(arg_T, CV_32FC1);
  /*
  [R | T]가 행단위로 들어가 있다.마지막은 스케일항
  R00 R01 R02 | T00
  R11 R11 R12 | T10       => [R00 R01 R02 T00 R11 R11 R12 T10 R20 R21 R22 T20 1]
  R20 R21 R22 | T20
  */

  this->_pf_camera_R_t[0] = arg_R.at<double>(0,0);
  this->_pf_camera_R_t[1] = arg_R.at<double>(0,1);
  this->_pf_camera_R_t[2] = arg_R.at<double>(0,2);
  this->_pf_camera_R_t[3] = arg_T.at<double>(0,0);
  this->_pf_camera_R_t[4] = arg_R.at<double>(1,0);
  this->_pf_camera_R_t[5] = arg_R.at<double>(1,1);
  this->_pf_camera_R_t[6] = arg_R.at<double>(1,2);
  this->_pf_camera_R_t[7] = arg_T.at<double>(1,0);
  this->_pf_camera_R_t[8] = arg_R.at<double>(2,0);
  this->_pf_camera_R_t[9] = arg_R.at<double>(2,1);
  this->_pf_camera_R_t[10] = arg_R.at<double>(2,2);
  this->_pf_camera_R_t[11] = arg_T.at<double>(2,0);
  this->_pf_camera_R_t[12] = 1.;
}