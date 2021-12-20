#include "NodeHandler.h"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

bool NodeHandler::Make_KeyFrame(Mat arg_KeyFrame)
{
  //이미지를 받아서 키프레임을 생성합니다. 
  KeyFrame kf_NewKeyFrame(this->int_CurrentFrameIdx-1,this->_mat_InstrisicParam,arg_KeyFrame);
  //키프레임 노드 생성및 정보를 등록합니다. 
  Mat des;
  vector<KeyPoint> kp_vector;
  int kp_size = this->_Get_NumberOfOrbFeature(arg_KeyFrame,des,kp_vector);
  //부모키프레임을 등록합니다.
  if(!this->_pt_KeyFrames.empty())//부모키프레임이 존재할 수 있다면(이미 하나정도 저장되어있다면)
  {
    kf_NewKeyFrame.Set_FatherKeyFrame(*(_pt_KeyFrames.end()-1));
    (*(_pt_KeyFrames.end()-1))->Set_ChildKeyFrame(&kf_NewKeyFrame);
  }

  //<--------------수정해야하는부분(키프레임에 맵포인트를 넣을 때는 한번에 넣을것, 
  //현재 키프레임에 것도 같은걸로 인식 및 knn간에 인덱스가 넘어버리는 문제가 있음
  //knn도 적당하게 임계값을 줄 수 있는 방법 찾을 것)
  //하나씩 추가하는게 아니라 동시에 knn돌려서 찾는게 제일 적합할 수도 있다. 비어있는지 확인하고 
  for(int idx_descriptor =0; idx_descriptor<des.rows; idx_descriptor++)
  {
    int int_DescriptorSize =32;
    Mat test_descriptor = des(Rect(0,idx_descriptor,int_DescriptorSize,1));
    MapPoint* return_MapPoint;//현재 descriptor를 통해서 맵포인트를 불러온다. 
    //해당 맵포인트가 없다면, 새로할당한다. 
    cout<<"키프레임에 삽입되는 맵포인트의 인덱스 : "<<idx_descriptor<<endl;
    Make_MapPoint(test_descriptor,return_MapPoint);
    //맵포인트에 현재 프레임을 포함시킵니다.
    return_MapPoint->vec_pt_ObservedKeyFrame.push_back(this->int_CurrentFrameIdx-1);
    //키프레임에서의 키포인트정보를 등록합니다.
    return_MapPoint->map_KeyPointInKeyFrame[this->int_CurrentFrameIdx-1] = kp_vector[idx_descriptor];
    //키프레임에서의 맵포인트의 인덱스 정보를 등록합니다.
    return_MapPoint->map_pointInd[this->int_CurrentFrameIdx-1] =idx_descriptor;
    //키프레임에 최종적으로 맵포인트의 pointer를 등록합니다. 
    kf_NewKeyFrame.Add_MapPoint(return_MapPoint);
  }
  
  
  return true;
}
bool NodeHandler::Is_GoodKeyFrame(Mat arg_candidateImage)
{
  // 초기 값이거나 20프레임이상 차이나는경우
  if (this->int_CurrentFrameIdx ==0 || 
  ((this->int_CurrentFrameIdx - this->int_LastKeyFrameIdx) >= 20))
  {
    this->int_CurrentFrameIdx+=1;
  }
  else
  {
    this->int_CurrentFrameIdx+=1;
    return false; 
  }  
  Mat des;
  vector<KeyPoint> kp_vector;
  int kp_size = this->_Get_NumberOfOrbFeature(arg_candidateImage,des,kp_vector);
  // 초기 값이거나 키포인트의 수가 50개이상인 경우
  if(this->int_CurrentFrameIdx ==0 || kp_size>50)
  {

  }
  else
  {
    return false;
  }
  //후보 프레임이 다른 키프레임보다 90프로 이하로 다른 점들을 트랙킹하는지 확인합니다. 
  //des의 행단위로 검사합니다.
  cout<<"현재 유지하고 있는 점의 수 "<<this->Get_MapPointSize()<<"마지막으로 넣은 키프레임 인덱스"<<this->int_LastKeyFrameIdx<<endl;
  vector<int> no_tracking_point;
  for(int idx_descriptor =0; idx_descriptor<des.rows; idx_descriptor++)
  {
    int int_DescriptorSize =32;
    Mat test_descriptor = des(Rect(0,idx_descriptor,int_DescriptorSize,1));
    bool Is_tp = this->_Is_TrackingMapPoint(test_descriptor);
    if(Is_tp)
    {
      
    }
    else
    {
      // cout<<"no"<<endl;
      no_tracking_point.push_back(idx_descriptor);
    }
  }
  // cout<<"처음 발견하는 점의 수 : "<<no_tracking_point.size()<<"  전체 점의 수 : "<<des.rows<<endl;
  if (no_tracking_point.size() <= (int)(des.rows*0.1))//10프로 이상의 점들이 처음보는 점들이어야함.
  {
    // cout<<"처음 발견하는 점의 수 : "<<no_tracking_point.size()<<"  전체 점의 수 : "<<des.rows<<endl;
    return false;
  }
  this->int_LastKeyFrameIdx = this->int_CurrentFrameIdx-1;
  return true;
}
bool NodeHandler::_Is_TrackingMapPoint(Mat arg_Descriptor)
{
  if(this->Get_MapPointSize() ==0)//만약에 맵포인트가 없다면 
  {
    return false;//아직은 추적하지 않은 맵포인트입니다.
  }
  else//만약에 맵포인트가 있다면 
  {
    MapPoint* mp_Old_MapPoint;
    if(this->_Is_NewMapPoint(arg_Descriptor,mp_Old_MapPoint))//새로운 맵포인트라면
    {
      return false;//아직은 추적하지 않은 맵포인트입니다.
    }
    else//맵포인트가 이전에 있던거라면 
    {
      return true;//추적중인 맵포인트입니다.
    }
  }
}

int NodeHandler::_Get_NumberOfOrbFeature(Mat arg_candidateImage, Mat&des, vector<KeyPoint>& kp)
{
  const static auto& _orb_OrbHandle = ORB::create();
  _orb_OrbHandle->detectAndCompute(arg_candidateImage,noArray(),kp,des);
  return kp.size();
}

bool NodeHandler::Make_MapPoint(Mat arg_Descriptor, MapPoint* &arg_OutputMapPoint)
{
  if(this->Get_MapPointSize() ==0)//만약에 맵포인트가 없다면 
  {
    return this->_Make_NewMapPoint(arg_Descriptor,arg_OutputMapPoint);//새로운 맵포인트 추가합니다.
  }
  else//만약에 맵포인트가 있다면 
  {
    MapPoint* mp_Old_MapPoint;
    if(this->_Is_NewMapPoint(arg_Descriptor,mp_Old_MapPoint))//새로운 맵포인트라면
    {
      return this->_Make_NewMapPoint(arg_Descriptor,arg_OutputMapPoint);//새로운 맵포인트 추가합니다.
    }
    else//맵포인트가 이전에 있던거라면 
    {
      arg_OutputMapPoint = mp_Old_MapPoint;//과거의 맵포인트를 출력합니다. 
      return true;
    }
  }
}

int NodeHandler::Get_MapPointSize(void)
{
  return this->_pt_MapPoints.size();
}

bool NodeHandler::_Make_NewMapPoint(Mat arg_Descriptor, MapPoint* &arg_OutputMapPoint)
{

  MapPoint* mp_newPoint = new MapPoint();
  mp_newPoint->mat_Orbdescirptor =arg_Descriptor;
  mp_newPoint->int_Node = this->_int_MapPointIdx++;
  arg_OutputMapPoint = mp_newPoint;
  
  if(this->Get_MapPointSize()==0)//초기 descriptor라면 
  {
    this->_full_descirptor = arg_Descriptor.clone();//초기 descriptor를 복사합니다. 
  }
  else//이미 descriptor가 할당되어있다면
  {
    this->_Apply_DescriptorMat(arg_Descriptor);//합쳐버립니다.
  }
  this->_pt_MapPoints.push_back(mp_newPoint);//새로운 맵포인트를 클래스내에 추가합니다. 
  
  return true;
}


bool NodeHandler::_Is_NewMapPoint(Mat arg_Descriptor, MapPoint* &arg_Old_MapPoint)
{
  vector< vector<DMatch>> matches;
  this->_match_OrbMatchHandle->knnMatch(arg_Descriptor,this->_full_descirptor,matches,2);//쿼리 디스크립터를 찾습니다. 
  sort(matches.begin(),matches.end());
  cout<<"match size : "<<matches.size()<<endl;
  cout<<"match 0 size : "<<matches[0].size()<<endl;

  const float ratio_thresh = 0.75f;
  if (matches[0][0].distance < ratio_thresh * matches[0][1].distance)
  {
      //임계값 이내라면 매칭 성공으로 판정
      int old_index = matches[0][0].trainIdx;//일치하는 인덱스 
      arg_Old_MapPoint = _pt_MapPoints[old_index];
      return false; //새로운 맵포인트가 아니다.
  }
  else//임계값 이하라서 새로운 매칭을 추가한다. 
  {
      return true;
  }
}

void NodeHandler::_Apply_DescriptorMat(Mat arg_Descriptor)
{
  int int_DescriptorSize =32;
  int int_ReturnRowSize = this->_full_descirptor.cols+1;

  Mat return_Descriptor(int_ReturnRowSize,int_DescriptorSize,CV_8SC1);
  this->_full_descirptor.copyTo(return_Descriptor(Rect(0,0,_full_descirptor.cols,int_DescriptorSize)));
  this->_full_descirptor.copyTo(return_Descriptor(Rect(0,_full_descirptor.rows,1,int_DescriptorSize)));
  this->_full_descirptor = return_Descriptor;//갱신
}