#include "NodeHandler.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "CameraTool.h"
#include <thread>

using namespace std;
using namespace cv;

NodeHandler::NodeHandler()
{

}
NodeHandler::NodeHandler(int arg_FrameThreshold)
{
  this->_int_FrameThreshold = arg_FrameThreshold;
}
NodeHandler::NodeHandler(int arg_FrameThreshold,int arg_LocalWindowSize)
{
  this->_int_FrameThreshold = arg_FrameThreshold;
  this->_int_LocalWindowSize = arg_LocalWindowSize;
}

bool NodeHandler::ValidateAndAddFrame(Mat arg_candidateImage)
{
  //step 1 : 프레임간의 차이가 적당한지 구한다. 
  if (this->int_CurrentFrameIdx ==0 || 
  ((this->int_CurrentFrameIdx - this->int_LastKeyFrameIdx) >= this->_int_FrameThreshold)){
    this->int_CurrentFrameIdx+=1;}
  else{
    this->int_CurrentFrameIdx+=1;
    return false; 
  }
  this->int_LastKeyFrameIdx = this->int_CurrentFrameIdx-1;
  return true;
}


int NodeHandler::_Get_NumberOfOrbFeature(Mat arg_candidateImage, Mat&des, vector<KeyPoint>& kp)
{
  const static auto& _orb_OrbHandle = ORB::create(500,1.2,8,31,0,2);
  _orb_OrbHandle->detectAndCompute(arg_candidateImage,noArray(),kp,des);
  return kp.size();
}

vector<KeyFrame*> NodeHandler::Get_LocalKeyFrame(void)
{
  return this->_pt_LocalWindowKeyFrames;
}



bool NodeHandler::IsNiceTime(void)
{
  if (this->int_CurrentFrameIdx ==0 || 
  ((this->int_CurrentFrameIdx - this->int_LastKeyFrameIdx) >= this->_int_FrameThreshold)){
    this->int_CurrentFrameIdx+=1;}
  else{
    this->int_CurrentFrameIdx+=1;
    return false; 
  }
  this->int_LastKeyFrameIdx = this->int_CurrentFrameIdx-1;
  return true;
}

bool NodeHandler::GetLastFrame(KeyFrame* &p_lastFrame)
{
  this->m_sharedlock.lock();
  int int_Current_LocalSize = this->_pt_LocalWindowKeyFrames.size();
  this->m_sharedlock.unlock();
  if(int_Current_LocalSize ==0){
    return false;}
  else{
    this->m_sharedlock.lock();
    p_lastFrame = this->_pt_LocalWindowKeyFrames.back();
    this->m_sharedlock.unlock();
    return true;}
}
bool NodeHandler::AddNewKeyFrame(KeyFrame* p_NewFrame)
{
  

  this->_pt_LocalWindowKeyFrames.push_back(p_NewFrame);//로컬 키프레임에 등록 
  this->_pt_KeyFrames.push_back(p_NewFrame);//전역 키프레임에 등록

  
  return true;
}
bool NodeHandler::SetImageFeature(KeyFrame* p_NewFrame, Mat Image)
{
  vector<KeyPoint> kp_vector;
  Mat des;
  int kp_size = this->_Get_NumberOfOrbFeature(Image,des,kp_vector);//현재 프레임에 대한 feature들을 얻어온다.

  //부모키프레임을 등록합니다.
  if(!this->_pt_KeyFrames.empty())//부모키프레임이 존재할 수 있다면(이미 하나정도 저장되어있다면)
  {
    p_NewFrame->Set_FatherKeyFrame(*(_pt_KeyFrames.end()-1));
    (*(_pt_KeyFrames.end()-1))->Set_ChildKeyFrame(p_NewFrame);
  }
  int int_DescriptorSize = 32;
  p_NewFrame->Set_Descriptor(des);//디스크립터 등록합니다.
  p_NewFrame->Set_KeyPoint(kp_vector);//키포인터 등록합니다. 
}
void NodeHandler::SetRt(KeyFrame* p_TempFrame, Mat R,Mat T)
{
  this->m_sharedlock.lock();
  p_TempFrame->Set_Rt(R,T);
  this->m_sharedlock.unlock();
}
void NodeHandler::GetRtParam(KeyFrame* p_TempFrame, float* &R_tparam)
{
  this->m_sharedlock.lock();
  p_TempFrame->Get_Rt(R_tparam);
  this->m_sharedlock.unlock();
}
vector<MapPoint*> NodeHandler::Get_localMapPoint()
{
  this->m_sharedlock.lock();
  auto return_vec = _pt_LocalWindowMapPoints;
  this->m_sharedlock.unlock();
  return return_vec;
}
void NodeHandler::Add_CandidateKeyFrame(NewKeyFrameSet* matches)
{
  this->m_sharedlock.lock();
  v_newKeyFrame.push_back(matches);//새롭게 키프레임으로 벡터에 넣어둡니다. 
  this->m_sharedlock.unlock();
}