#include "NodeHandlerWithStatic.h"
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
void NodeHandler::Delete_Descriptor(int arg_delIdx)
{
  int int_LocalReturnRowSize = this->_mat_LocalDescriptor.rows-1;
  int int_DescriptorSize =32;
  Mat return_Descriptor(int_LocalReturnRowSize,int_DescriptorSize,CV_8UC1);
  if(arg_delIdx !=0)
  {
    this->_mat_LocalDescriptor(Rect(0,0,int_DescriptorSize,arg_delIdx)).copyTo(return_Descriptor(Rect(0,0,int_DescriptorSize,arg_delIdx)));
  }
  this->_mat_LocalDescriptor(Rect(0,arg_delIdx+1,int_DescriptorSize,int_LocalReturnRowSize-arg_delIdx)).copyTo(return_Descriptor(Rect(0,arg_delIdx,int_DescriptorSize,int_LocalReturnRowSize-arg_delIdx)));
  this->_mat_LocalDescriptor = return_Descriptor;//갱신
}
void NodeHandler::Change_Window(KeyFrame* arg_NewKeyFrame)
{
  //일단 사이즈를 확인한다.
  int int_Current_LocalSize = this->_pt_LocalWindowKeyFrames.size();
  if(int_Current_LocalSize>=20)
  {
    this->_pt_LocalWindowKeyFrames.push_back(arg_NewKeyFrame);
    //맨앞의 키프레임을 제거해야한다.
    //해당 키프레임들에 해당하는 맵포인트들을 재구성한다. 
    KeyFrame* kfp_ErasedKeyFrame = *_pt_LocalWindowKeyFrames.begin();
    _pt_LocalWindowKeyFrames.erase(_pt_LocalWindowKeyFrames.begin());//처음원소를 제거합니다. 
    vector<MapPoint*> mps_ErasedMapPoint = kfp_ErasedKeyFrame->Get_MapPoint();
    int minimum_keyframe_index = int_CurrentFrameIdx-1-this->_int_LocalWindowSize;//현재 프레임 값에서 윈도우 사이즈 만큼 뺍니다. 
    int delete_count =0;
    for(int i=0; i<mps_ErasedMapPoint.size(); i++)
    {
      vector<int> checked_Frame = (mps_ErasedMapPoint[i])->vec_pt_ObservedKeyFrame;
      bool Is_using= false;
      int using_Frame_number =0;
      for(int j=0; j<checked_Frame.size(); j++)
      {
        if(checked_Frame[j]>=minimum_keyframe_index)
        {
          Is_using =true;
          using_Frame_number = checked_Frame[j];
          break;
        }
      }

      
      if(!Is_using)
      {//사용하고 있지 않다면 LocalMapPoint에서 제거합니다. _mat_LocalDescriptor에서도 제거합니다. 
        // this->_pt_LocalWindowMapPoints[]//로컬맵포인트에서 사용중인 위치를 알아야하는데 이걸 어떻게 알까...찾는함수를 써볼까.
        auto &lmp = this->_pt_LocalWindowMapPoints;
        auto it = find(lmp.begin(),lmp.end(),mps_ErasedMapPoint[i]);
        if(it !=lmp.end())
        {
          int delete_index = it-lmp.begin();//지울 인덱스 번호 
          lmp.erase(lmp.begin()+delete_index);//안쓰이는 맵포인트를 제거 
          Delete_Descriptor(delete_index);//해당 인덱스에 해당하는 맵포인트를 제거합니다...
          delete_count++;
        }
      }
      else{
        if(i<3)
        {
          // for(int k=0; k<3; k++)
          // {
              
          // }
          // cout<<"해당 맵포인트가 관측되는 프레임의 갯수"<<checked_Frame.size()<<endl;//관측되는 프레임 수 
          // cout<<"아직 사용중입니다. 사용중인 프레임이름"<<using_Frame_number<<endl;
        }
        
      }
    }
    // cout<<"지워지는 프레임 번호 : "<<kfp_ErasedKeyFrame->Get_KeyIndex()<<endl;
    // cout<<"현재 최소 프레임 :"<<minimum_keyframe_index<<", 지워지는 맵포인트 수 : "<<delete_count<<endl;

    
  }
  else
  {
    _pt_LocalWindowKeyFrames.push_back(arg_NewKeyFrame);
    // for(auto tet =this->_pt_LocalWindowKeyFrames.begin(); tet !=_pt_LocalWindowKeyFrames.end();tet++ )
    // {
    //   cout<<"프레임 번호 : "<<(*tet)->Get_KeyIndex()<<endl;
    // }
  }
}
bool NodeHandler::Make_MapPoint_pix2pixMatch(Mat arg_descriptor,vector<KeyPoint> arg_KeyPoint, KeyFrame* arg_KeyFrame)
{
  int int_DescriptorSize = 32;
  if(this->Get_MapPointSize()==0)
  //초기 descriptor라면, 전부 추가하고, 맵포인트 만들고, 키프레임에 등록한다.
  {
    for(int idx_descriptor =0; idx_descriptor<arg_descriptor.rows; idx_descriptor++)
    {
      MapPoint* mp_newPoint = new MapPoint();
      Mat mat_newDescriptor = arg_descriptor(Rect(0,idx_descriptor,int_DescriptorSize,1));
      mp_newPoint->mat_Orbdescriptor =mat_newDescriptor;
      mp_newPoint->int_Node = this->_int_MapPointIdx++;
      // arg_OutputMapPoint = mp_newPoint;
      
      //전체 descriptor를 맵포인트에 포함시킵니다. 
      if(this->Get_MapPointSize()==0){
        this->_mat_FullDescriptor = mat_newDescriptor.clone();//초기 descriptor를 복사합니다. 
        this->_mat_LocalDescriptor = mat_newDescriptor.clone();
      }
      else{
        this->_Apply_DescriptorMat(mat_newDescriptor);//합쳐버립니다.
      }
      

      this->_pt_MapPoints.push_back(mp_newPoint);//새로운 맵포인트를 클래스내에 추가합니다.
      this->_pt_LocalWindowMapPoints.push_back(mp_newPoint);//새로운 맵포인트를 클래스내에 추가합니다.
      mp_newPoint->vec_pt_ObservedKeyFrame.push_back(this->int_CurrentFrameIdx-1);
      //키프레임에서의 전역 키포인트정보를 등록합니다.
      mp_newPoint->map_KeyPointInKeyFrame[this->int_CurrentFrameIdx-1] = arg_KeyPoint[idx_descriptor];
      //전역 키프레임에서의 맵포인트의 인덱스 정보를 등록합니다.
      mp_newPoint->map_pointInd[this->int_CurrentFrameIdx-1] =idx_descriptor;
      //현재 키프레임에 최종적으로 맵포인트의 pointer를 등록합니다. 
      arg_KeyFrame->Add_MapPoint(mp_newPoint); 
    }
  }
  else
  {
    vector< vector<DMatch>> matches;//매치를 저장할 변수입니다.
    this->_match_OrbMatchHandle->knnMatch(arg_descriptor,this->_mat_LocalDescriptor,matches,2);//쿼리 디스크립터를 찾습니다. 
    sort(matches.begin(),matches.end());
    vector<DMatch> good_matches;
    const float ratio_thresh = 0.85f;//
    int coincide_size = 0;
    for (size_t idx_descriptor = 0; idx_descriptor < matches.size(); idx_descriptor++)
    {
      DMatch targetMatch = matches[idx_descriptor][0];
      MapPoint* pt_targetPoint;
      if (matches[idx_descriptor][0].distance < ratio_thresh * matches[idx_descriptor][1].distance)//기존의 맵포인트를 찾아서 키프레임에 등록합니다.
      {
        coincide_size++;
        pt_targetPoint = this->_pt_LocalWindowMapPoints[targetMatch.trainIdx];//이거는 로컬에서의 맵포인트 인덱스이다. 기존과는 다르게
        
      }
      else//새로운 맵포인트를 만들어서 등록합니다. 
      {
        pt_targetPoint = new MapPoint();
        Mat mat_newDescriptor = arg_descriptor(Rect(0,idx_descriptor,int_DescriptorSize,1));
        pt_targetPoint->mat_Orbdescriptor =mat_newDescriptor;
        pt_targetPoint->int_Node = this->_int_MapPointIdx++;
        //전체 descriptor를 맵포인트에 포함시킵니다. 
        if(this->Get_MapPointSize()==0){
          this->_mat_FullDescriptor = mat_newDescriptor.clone();//초기 descriptor를 복사합니다. 
          this->_mat_LocalDescriptor = mat_newDescriptor.clone();
        }
        else{
          this->_Apply_DescriptorMat(mat_newDescriptor);//합쳐버립니다.
        }

        this->_pt_MapPoints.push_back(pt_targetPoint);//새로운 맵포인트를 클래스내에 추가합니다.
        this->_pt_LocalWindowMapPoints.push_back(pt_targetPoint);//새로운 맵포인트를 클래스내에 추가합니다.
      }
      //새롭게 만들었던 안만들었든 추가로 등록합니다.
      pt_targetPoint->vec_pt_ObservedKeyFrame.push_back(this->int_CurrentFrameIdx-1);
      //키프레임에서의 키포인트정보를 등록합니다.
      pt_targetPoint->map_KeyPointInKeyFrame[this->int_CurrentFrameIdx-1] = arg_KeyPoint[idx_descriptor];
      //키프레임에서의 맵포인트의 인덱스 정보를 등록합니다.
      pt_targetPoint->map_pointInd[this->int_CurrentFrameIdx-1] =idx_descriptor;
      //키프레임에 최종적으로 맵포인트의 pointer를 등록합니다. 
      arg_KeyFrame->Add_MapPoint(pt_targetPoint); 
    }
  }
  Change_Window(arg_KeyFrame);
}

bool NodeHandler::Make_KeyFrame(Mat arg_KeyFrame)
{
  //이미지를 받아서 키프레임을 생성합니다. 
  
  this->_pt_KeyFrames.push_back(new KeyFrame(this->int_CurrentFrameIdx-1, this->_mat_InstrisicParam));
  KeyFrame* kfp_NewKeyFrame = *(this->_pt_KeyFrames.end()-1);
  //키프레임 노드 생성및 정보를 등록합니다. 
  Mat des;
  
  vector<KeyPoint> kp_vector;
  int kp_size = this->_Get_NumberOfOrbFeature(arg_KeyFrame,des,kp_vector);
  //부모키프레임을 등록합니다.
  if(!this->_pt_KeyFrames.empty())//부모키프레임이 존재할 수 있다면(이미 하나정도 저장되어있다면)
  {
    kfp_NewKeyFrame->Set_FatherKeyFrame(*(_pt_KeyFrames.end()-1));
    (*(_pt_KeyFrames.end()-1))->Set_ChildKeyFrame(kfp_NewKeyFrame);
  }

  //<--------------수정해야하는부분(키프레임에 맵포인트를 넣을 때는 한번에 넣을것, 
  //현재 키프레임에 것도 같은걸로 인식 및 knn간에 인덱스가 넘어버리는 문제가 있음
  //knn도 적당하게 임계값을 줄 수 있는 방법 찾을 것)
  //하나씩 추가하는게 아니라 동시에 knn돌려서 찾는게 제일 적합할 수도 있다. 비어있는지 확인하고 
  this->Make_MapPoint_pix2pixMatch(des,kp_vector,kfp_NewKeyFrame);
    
  return true;
}
bool NodeHandler::Is_GoodKeyFrame(Mat arg_candidateImage)
{
  // 초기 값이거나 20프레임이상 차이나는경우
  if (this->int_CurrentFrameIdx ==0 || 
  ((this->int_CurrentFrameIdx - this->int_LastKeyFrameIdx) >= this->_int_FrameThreshold))
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
  int kp_size = this->_Get_NumberOfOrbFeature(arg_candidateImage,des,kp_vector);//------------------------강인한 특징만 뽑기 
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
  if (no_tracking_point.size() <= (int)(des.rows*0.1))//10프로 이상의 점들이 처음보는 점들이어야함.
  {
    // cout<<"처음 발견하는 점의 수 : "<<no_tracking_point.size()<<"  전체 점의 수 : "<<des.rows<<endl;
    return false;
  }
  cout<<"현재 유지하고 있는 점의 수 "<<this->_pt_LocalWindowMapPoints.size()<<" tracking하는 점의 수 :  "<<500-no_tracking_point.size()<<" 마지막으로 넣은 키프레임 인덱스"<<this->int_LastKeyFrameIdx<<endl;
  
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
  mp_newPoint->mat_Orbdescriptor =arg_Descriptor;
  mp_newPoint->int_Node = this->_int_MapPointIdx++;
  arg_OutputMapPoint = mp_newPoint;
  
  if(this->Get_MapPointSize()==0)//초기 descriptor라면 
  {
    this->_mat_FullDescriptor = arg_Descriptor.clone();//초기 descriptor를 복사합니다. 
    this->_mat_LocalDescriptor = arg_Descriptor.clone();
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
  // cout<<"arg type"<<type2str(arg_Descriptor.type())<<endl;
  this->_match_OrbMatchHandle->knnMatch(arg_Descriptor,this->_mat_LocalDescriptor,matches,2);//쿼리 디스크립터를 찾습니다. 
  sort(matches.begin(),matches.end());
  // cout<<"match size : "<<matches.size()<<endl;
  // cout<<"match 0 size : "<<matches[0].size()<<endl;

  const float ratio_thresh = 0.75f;
  if (matches[0][0].distance < ratio_thresh * matches[0][1].distance)
  {//최근접 거리 비율 전략
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
  //전역 갱신 
  int int_DescriptorSize =32;
  int int_ReturnRowSize = this->_mat_FullDescriptor.rows+1;
  Mat return_Descriptor(int_ReturnRowSize,int_DescriptorSize,CV_8UC1);
  this->_mat_FullDescriptor.copyTo(return_Descriptor(Rect(0,0,int_DescriptorSize,_mat_FullDescriptor.rows)));
  arg_Descriptor.copyTo(return_Descriptor(Rect(0,_mat_FullDescriptor.rows,int_DescriptorSize,1)));
  this->_mat_FullDescriptor = return_Descriptor;//갱신
  
  //윈도우 단위로 갱신
  int int_LocalReturnRowSize = this->_mat_LocalDescriptor.rows+1;
  Mat Local_return_Descriptor(int_LocalReturnRowSize,int_DescriptorSize,CV_8UC1);
  this->_mat_LocalDescriptor.copyTo(Local_return_Descriptor(Rect(0,0,int_DescriptorSize,this->_mat_LocalDescriptor.rows)));
  arg_Descriptor.copyTo(Local_return_Descriptor(Rect(0,this->_mat_LocalDescriptor.rows,int_DescriptorSize,1)));
  this->_mat_LocalDescriptor = Local_return_Descriptor;//갱신
}