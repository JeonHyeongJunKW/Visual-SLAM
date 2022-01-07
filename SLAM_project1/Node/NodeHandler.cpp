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

  // //step 2 : 가지고 있는 키포인트의 개수를 구합니다. 
  
  // KeyFrame* kfp_TempFrame = new KeyFrame(this->int_CurrentFrameIdx-1, this->_mat_InstrisicParam);//해당 키프레임 포인터 가져오고
  // //키프레임 노드 생성및 정보를 등록합니다. 

  // Mat des;
  // vector<KeyPoint> kp_vector;
  // int kp_size = this->_Get_NumberOfOrbFeature(arg_candidateImage,des,kp_vector);//현재 프레임에 대한 feature들을 얻어온다.

  // //부모키프레임을 등록합니다.
  // if(!this->_pt_KeyFrames.empty())//부모키프레임이 존재할 수 있다면(이미 하나정도 저장되어있다면)
  // {
  //   kfp_TempFrame->Set_FatherKeyFrame(*(_pt_KeyFrames.end()-1));
  //   (*(_pt_KeyFrames.end()-1))->Set_ChildKeyFrame(kfp_TempFrame);
  // }
  // int int_DescriptorSize = 32;
  // kfp_TempFrame->Set_Descriptor(des);//디스크립터 등록합니다.
  // kfp_TempFrame->Set_KeyPoint(kp_vector);//키포인터 등록합니다. 

  // Change_Window(kfp_TempFrame);//로컬윈도우를 바꾸면서 매칭을 다시만듭니다. 

  this->int_LastKeyFrameIdx = this->int_CurrentFrameIdx-1;
  return true;
}




bool NodeHandler::Change_Window(KeyFrame* arg_NewKeyFrame)
{
  const int HOMO =0;
  const int FUND =1;
  int int_DescriptorSize = 32;
  int int_Current_LocalSize = this->_pt_LocalWindowKeyFrames.size();
  //아무것도 없는경우. 
  int max_match_idx = -1;
  int max_match = 0;

  
  vector<KeyFrame*> tempFrame;//확인용 프레임입니다.
  vector<Match_Set*> tempMatch;//확인용 매칭입니다. 
  int die_size = 0;
  cout<<"----------------------------"<<endl;
  if(int_Current_LocalSize ==0){//초기에 아무프레임도 없다면
    this->_pt_LocalWindowKeyFrames.push_back(arg_NewKeyFrame);//로컬 키프레임에 등록 
    this->_pt_KeyFrames.push_back(arg_NewKeyFrame);//전역 키프레임에 등록
    cout<<"초기 프레임을 삽입합니다."<<endl;
    return true;}//처음에는 맵포인트가 없기때문에 등록하지 않습니다. 
  else{//초기 프레임이라도 있다면 
    KeyFrame* lastFrame = this->_pt_LocalWindowKeyFrames.back();
    
    vector<vector<DMatch>> matches;//매치를 저장할 변수입니다.
    this->_match_OrbMatchHandle->knnMatch(arg_NewKeyFrame->Get_Descriptor(),lastFrame->Get_Descriptor(),matches,2);//쿼리 디스크립터를 찾습니다. 
    //initial pose estimation from last frame
    vector<DMatch> good_matches;//좋은 매치만 저장됩니다.
    const float ratio_thresh = 0.65f;

    for (size_t i = 0; i < matches.size(); i++){
      if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)//query -> first
      {
        good_matches.push_back(matches[i][0]);
        tempMatch.push_back(new Match_Set(arg_NewKeyFrame,
                                        lastFrame,
                                        &(arg_NewKeyFrame->Get_keyPoint()[i]),
                                        &(lastFrame->Get_keyPoint()[matches[i][0].trainIdx])));
      }
    }
    if(good_matches.size()<5)//현재 로컬프레임과의 매칭이 잘되었는지 확인합니다. 
    {
      cout<<"relocalization start"<<endl;
      cout <<"처음 키프레임과 나중키프레임이 연결이 안되었습니다."<<endl;
      exit(0);
    }

    vector<Point2f> current_point;
    vector<Point2f> last_point;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      Match_Set* homo_set = tempMatch[i];
      current_point.push_back(homo_set->kp_first->pt);//현재프레임
      last_point.push_back(homo_set->kp_second->pt);//나중프레임
    }

    thread homocheck;
    thread fundamentalcheck;

    float sh;
    Mat Homography;
    float sf;
    Mat FundamentalMat;
    vector<int> h_inlier;
    vector<int> f_inlier;
    homocheck = thread(CheckHomography,current_point,last_point,&sh,&Homography,&h_inlier);
    fundamentalcheck= thread(CheckFundamental, current_point,last_point,lastFrame->Get_IntrinsicParam(),&sf,&FundamentalMat,&f_inlier);
    homocheck.join();
    fundamentalcheck.join();
    Mat return_R;
    Mat return_T;
    vector<Point2f> inliner_current_point;
    vector<Point2f> inliner_reference_point;
    int mode;
    vector<int> inlier_set;
    if(sh/(sh+sf) >0.45)
    {
      mode = HOMO;
      inlier_set = h_inlier;
      cout<<"model is homography"<<endl; 
      // cout<<"인라이어의 수 : "<<inlier_set.size()<<endl;
    }
    else
    {
      mode = FUND;
      inlier_set = f_inlier;
      cout<<"model is Fundamental"<<endl;
      // cout<<"인라이어의 수 : "<<inlier_set.size()<<endl;
    }
    for(int j=0; j<inlier_set.size(); j++)
    {
      inliner_current_point.push_back(current_point[inlier_set[j]]);
      inliner_reference_point.push_back(last_point[inlier_set[j]]);
    }

    if(mode ==FUND)
    {
      ValidateHomographyRt(inliner_current_point,inliner_reference_point,lastFrame->Get_IntrinsicParam(),return_R,return_T);
    }
    else
    {
      ValidateFundamentalRt(inliner_current_point,inliner_reference_point,lastFrame->Get_IntrinsicParam(),return_R,return_T);
    }
    cout<<return_R<<endl;
    cout<<return_T<<endl;
    exit(0);
    
    for(int int_keyIdx =0; int_keyIdx<this->_pt_LocalWindowKeyFrames.size(); int_keyIdx++)
    {//키프레임을 비교합니다. 
      vector<vector<DMatch>> matches;//매치를 저장할 변수입니다.
      if(int_Current_LocalSize ==1)//아직 하나밖에 없는경우
      {//기존의 것과 knn으로 비교합니다.
        this->_match_OrbMatchHandle->knnMatch(arg_NewKeyFrame->Get_Descriptor(),this->_pt_LocalWindowKeyFrames[int_keyIdx]->Get_Descriptor(),matches,2);//쿼리 디스크립터를 찾습니다. 
        const float ratio_thresh = 0.65f;
        for (size_t i = 0; i < matches.size(); i++){
          if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)//query -> first
          {
            good_matches.push_back(matches[i][0]);
            tempMatch.push_back(new Match_Set(arg_NewKeyFrame,
                                            _pt_LocalWindowKeyFrames[int_keyIdx],
                                            &(arg_NewKeyFrame->Get_keyPoint()[i]),
                                            &(_pt_LocalWindowKeyFrames[int_keyIdx]->Get_keyPoint()[matches[i][0].trainIdx])));
          }
        }
        if(good_matches.size()>0)//현재 로컬프레임과의 매칭이 잘되었는지 확인합니다. 
        {
          tempFrame.push_back(_pt_LocalWindowKeyFrames[int_keyIdx]);
          if(good_matches.size() > max_match)
          {
            max_match = good_matches.size();
            max_match_idx = int_keyIdx;
          }
        }else{
          cout <<"처음 키프레임과 나중키프레임이 연결이 안되었습니다."<<endl;
          exit(0);
          }
      }//하나만 추가되는 경우 
      else//일반적인 케이스
      {//이전의 맵포인트로 얻은거를 기반으로해서 얻습니다.

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
      if(good_matches.size()>50){
        if(good_matches.size()>270){
          cout <<"reference랑 너무 많이 연결이 되었습니다. "<<endl;
        }else{
          cout <<"reference랑 연결이 되었습니다. "<<endl;
        }
      }
      cout<<"reference frame("<<max_match_idx<<")을 정하고 Visual Odometry를 구합니다."<<endl;
      cout<<"매칭의 개수 : "<<tempMatch.size()<<endl;

      //임시 프레임과 매칭을 갱신합니다.
      // tempFrame.push_back(arg_NewKeyFrame);
      
      /*
      1. homography나 Essential matrix를 써서 두 이미지간에 r,t,n값을 알아낸다. 
      2. 이를 기반으로하여 odometry의 초기값을 알아낸다. (맵포인트들은 나중에 삼각법으로 3차원상의 점으로 만들어서 구한다(local mapping에서 처리되는부분), 
          위치부터 트랙킹부터 한다. )
      3. 나중에 얻은 맵포인트들을 기반으로하여 현재 프레임에대한 평가를 내리고, 50개의 맵포인트가 트랙킹되는지 구한다.
      4. 또한 래펀런스와 유사한지도 확인한다. 이를 기반으로 추정한다. 
      */
      //Homography를 구합니다.
      
      exit(0);
      
      this->_pt_LocalWindowKeyFrames.clear();
      this->_pt_LocalWindowKeyFrames = tempFrame;
      this->_local_MatchSet.clear();
      this->_local_MatchSet = tempMatch;

      this->_pt_KeyFrames.push_back(arg_NewKeyFrame);//전역 키프레임에 등록
      
      // cout<<"K1 : "<<this->_pt_LocalWindowKeyFrames.size()<< "max match : "<< max_match<<endl;
      if(max_match<5)
      {
        // cout<<"K1 : "<<this->_pt_LocalWindowKeyFrames.size()<< "max match : "<< max_match<<endl;
        cout<<"small MapPoint : "<<max_match<<endl;
      }
      // cout<<"버려진 키프레임의 수"<<die_size<<endl;
      return true;
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
  this->m_sharedlock.lock();

  this->_pt_LocalWindowKeyFrames.push_back(p_NewFrame);//로컬 키프레임에 등록 
  this->_pt_KeyFrames.push_back(p_NewFrame);//전역 키프레임에 등록

  this->m_sharedlock.unlock();
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