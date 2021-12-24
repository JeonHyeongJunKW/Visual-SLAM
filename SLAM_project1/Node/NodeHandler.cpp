#include "NodeHandler.h"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
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

  //step 2 : 가지고 있는 키포인트의 개수를 구합니다. 
  
  KeyFrame* kfp_NewKeyFrame = new KeyFrame(this->int_CurrentFrameIdx-1, this->_mat_InstrisicParam);//해당 키프레임 포인터 가져오고
  //키프레임 노드 생성및 정보를 등록합니다. 

  Mat des;
  vector<KeyPoint> kp_vector;
  int kp_size = this->_Get_NumberOfOrbFeature(arg_candidateImage,des,kp_vector);
  //부모키프레임을 등록합니다.
  if(!this->_pt_KeyFrames.empty())//부모키프레임이 존재할 수 있다면(이미 하나정도 저장되어있다면)
  {
    kfp_NewKeyFrame->Set_FatherKeyFrame(*(_pt_KeyFrames.end()-1));
    (*(_pt_KeyFrames.end()-1))->Set_ChildKeyFrame(kfp_NewKeyFrame);
  }
  int int_DescriptorSize = 32;
  kfp_NewKeyFrame->Set_Descriptor(des);//디스크립터 등록합니다.
  kfp_NewKeyFrame->Set_KeyPoint(kp_vector);//키포인터 등록합니다. 

  Change_Window(kfp_NewKeyFrame);//로컬윈도우를 바꾸면서 매칭을 다시만듭니다. 

  this->int_LastKeyFrameIdx = this->int_CurrentFrameIdx-1;
  return true;
}




bool NodeHandler::Change_Window(KeyFrame* arg_NewKeyFrame)
{
  int int_DescriptorSize = 32;
  int int_Current_LocalSize = this->_pt_LocalWindowKeyFrames.size();
  //아무것도 없는경우. 
  int max_match_idx = -1;
  int max_match = 0;

  
  vector<KeyFrame*> tempFrame;//확인용 프레임입니다.
  vector<Match_Set*> tempMatch;//확인용 매칭입니다. 
  int die_size = 0;
  // cout<<"----------------------------"<<endl;
  if(int_Current_LocalSize ==0)
  {
    _pt_LocalWindowKeyFrames.push_back(arg_NewKeyFrame);
    return true;
  }
  else
  {
    //현재 키프레임이 가지고 있는 맵포인트와 로컬에 있는 맵포인트를 비교하여 트렉킹하는 점이 있는지 확인합니다. 
    //일단 어느정도 키프레임 단위로 키핑하고, 비교해서 있는지 판단, 개수 판단
    // cout<<"already have keyframe"<<endl;
    for(int int_keyIdx =0; int_keyIdx<this->_pt_LocalWindowKeyFrames.size(); int_keyIdx++)
    {//키프레임을 비교합니다. 
      vector< vector<DMatch>> matches;//매치를 저장할 변수입니다.
      //1:1 매칭을 통해서 비교합니다.
      this->_match_OrbMatchHandle->knnMatch(arg_NewKeyFrame->Get_Descriptor(),this->_pt_LocalWindowKeyFrames[int_keyIdx]->Get_Descriptor(),matches,2);//쿼리 디스크립터를 찾습니다. 
      vector<DMatch> good_matches;
      const float ratio_thresh = 0.65f;
      for (size_t i = 0; i < matches.size(); i++)
      {
          if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
          {
              good_matches.push_back(matches[i][0]);
              //바로 로컬 상관관계를 등록합니다. 
              tempMatch.push_back(new Match_Set(arg_NewKeyFrame,
                                              _pt_LocalWindowKeyFrames[int_keyIdx],
                                              &(arg_NewKeyFrame->Get_keyPoint()[i]),
                                              &(_pt_LocalWindowKeyFrames[int_keyIdx]->Get_keyPoint()[matches[i][0].trainIdx])));
          }
      }
      if(good_matches.size()>0)
      {
        if(good_matches.size()>max_match)
        {
          max_match = good_matches.size();
          max_match_idx = int_keyIdx;
        }
        //F_1에 추가해버립니다. 
        tempFrame.push_back(_pt_LocalWindowKeyFrames[int_keyIdx]);
        // cout<<_pt_LocalWindowKeyFrames[int_keyIdx]->Get_KeyIndex()<<"번째 포인터 : "<<good_matches.size()<<endl;
      }
      else
      {
        die_size++; //버려진 키프레임의 수
      }
    }
    //매칭되는게 없으면 다시 로컬라이제이션 해줍니다. 
    if(max_match ==0)
    {
      cout<<"relocalization start"<<endl;
    }
    else//있다면 그중에서 가장 큰걸 Reference Frame으로 잡고 개수 비교를 해서 추가할지 말지를 정합니다. 
    {
      KeyFrame* kfp_ReferenceFrame = _pt_LocalWindowKeyFrames[max_match_idx];
      if (max_match >= (int)(arg_NewKeyFrame->Get_keyPoint().size()*0.9))//10프로 이상의 점들이 처음보는 점들이어야함.
      { 
        cout<<"static type"<<endl;
        return false;
      }
      else
      {
        //임시 프레임과 매칭을 갱신합니다.
        tempFrame.push_back(arg_NewKeyFrame);
        
        this->_pt_LocalWindowKeyFrames.clear();
        this->_pt_LocalWindowKeyFrames = tempFrame;
        this->_local_MatchSet.clear();
        this->_local_MatchSet = tempMatch;

        this->_pt_KeyFrames.push_back(arg_NewKeyFrame);//전역 키프레임에 등록
        
        // cout<<"K1 : "<<this->_pt_LocalWindowKeyFrames.size()<< "max match : "<< max_match<<endl;
        if(max_match<5)
        {
          cout<<"K1 : "<<this->_pt_LocalWindowKeyFrames.size()<< "max match : "<< max_match<<endl;
        }
        // cout<<"버려진 키프레임의 수"<<die_size<<endl;
        return true;
      }
    }
  }
}

int NodeHandler::_Get_NumberOfOrbFeature(Mat arg_candidateImage, Mat&des, vector<KeyPoint>& kp)
{
  const static auto& _orb_OrbHandle = ORB::create();
  _orb_OrbHandle->detectAndCompute(arg_candidateImage,noArray(),kp,des);
  return kp.size();
}
vector<KeyFrame*> NodeHandler::Get_LocalKeyFrame(void)
{
  return this->_pt_LocalWindowKeyFrames;
}