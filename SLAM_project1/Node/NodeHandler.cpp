#include "NodeHandler.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "CameraTool.h"

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
  
  KeyFrame* kfp_TempFrame = new KeyFrame(this->int_CurrentFrameIdx-1, this->_mat_InstrisicParam);//해당 키프레임 포인터 가져오고
  //키프레임 노드 생성및 정보를 등록합니다. 

  Mat des;
  vector<KeyPoint> kp_vector;
  int kp_size = this->_Get_NumberOfOrbFeature(arg_candidateImage,des,kp_vector);//현재 프레임에 대한 feature들을 얻어온다.
  //부모키프레임을 등록합니다.
  if(!this->_pt_KeyFrames.empty())//부모키프레임이 존재할 수 있다면(이미 하나정도 저장되어있다면)
  {
    kfp_TempFrame->Set_FatherKeyFrame(*(_pt_KeyFrames.end()-1));
    (*(_pt_KeyFrames.end()-1))->Set_ChildKeyFrame(kfp_TempFrame);
  }
  int int_DescriptorSize = 32;
  kfp_TempFrame->Set_Descriptor(des);//디스크립터 등록합니다.
  kfp_TempFrame->Set_KeyPoint(kp_vector);//키포인터 등록합니다. 

  Change_Window(kfp_TempFrame);//로컬윈도우를 바꾸면서 매칭을 다시만듭니다. 

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
  cout<<"----------------------------"<<endl;
  if(int_Current_LocalSize ==0)
  {
    this->_pt_LocalWindowKeyFrames.push_back(arg_NewKeyFrame);//로컬 키프레임에 등록 
    this->_pt_KeyFrames.push_back(arg_NewKeyFrame);//전역 키프레임에 등록
    cout<<"초기 프레임을 삽입합니다."<<endl;
    //처음에는 맵포인트가 없기때문에 등록하지 않습니다. 
    return true;
  }
  else
  {
    //현재 키프레임이 가지고 있는 맵포인트와 로컬에 있는 맵포인트를 비교하여 트렉킹하는 점이 있는지 확인합니다. 
    //일단 어느정도 키프레임 단위로 키핑하고, 비교해서 있는지 판단, 개수 판단
    // cout<<"already have keyframe"<<endl;
    vector<DMatch> good_matches;
    for(int int_keyIdx =0; int_keyIdx<this->_pt_LocalWindowKeyFrames.size(); int_keyIdx++)
    {//키프레임을 비교합니다. 
      vector< vector<DMatch>> matches;//매치를 저장할 변수입니다.
      //1:1 매칭을 통해서 비교합니다.
      if(int_Current_LocalSize ==1)//아직 하나밖에 없는경우
      {//기존의 것과 knn으로 비교합니다.
        this->_match_OrbMatchHandle->knnMatch(arg_NewKeyFrame->Get_Descriptor(),this->_pt_LocalWindowKeyFrames[int_keyIdx]->Get_Descriptor(),matches,2);//쿼리 디스크립터를 찾습니다. 
        const float ratio_thresh = 0.65f;
        //일단 둘사이의 연관점을 기반으로 맵포인트를 만들어서 odometry를사용해야합니다. 
        for (size_t i = 0; i < matches.size(); i++)
        {
          if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
          {
            good_matches.push_back(matches[i][0]);
            //바로 로컬 상관관계를 등록합
            tempMatch.push_back(new Match_Set(arg_NewKeyFrame,
                                            _pt_LocalWindowKeyFrames[int_keyIdx],
                                            &(arg_NewKeyFrame->Get_keyPoint()[i]),
                                            &(_pt_LocalWindowKeyFrames[int_keyIdx]->Get_keyPoint()[matches[i][0].trainIdx])));
          }
        }
        if(good_matches.size()>0)
        {
          cout <<"처음 키프레임과 나중 키프레임이 연결이 되었습니다."<<endl;
          if(good_matches.size()>50)
          {
            if(good_matches.size()>270)
            {
              cout <<"reference랑 너무 많이 연결이 되었습니다. "<<endl;
            }
            else
            {
              cout <<"reference랑 연결이 되었습니다. "<<endl;
              max_match = good_matches.size();
              max_match_idx = int_keyIdx;
            }
            
          }
        }
        else
        {
          cout <<"처음 키프레임과 나중키프레임이 연결이 안되었습니다."<<endl;
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
      cout<<"reference frame("<<max_match_idx<<")을 정했고 Visual Odometry를 구합니다."<<endl;
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
      vector<Point2f> first_point;
      vector<Point2f> second_point;

      for( size_t i = 0; i < good_matches.size(); i++ )
      {
          //-- Get the keypoints from the good matches
          // first_point.push_back( kfp_ReferenceFrame->Get_keyPoint()[ good_matches[i].queryIdx ].pt );
          // second_point.push_back( arg_NewKeyFrame->Get_keyPoint()[ good_matches[i].trainIdx ].pt );
          Match_Set* homo_set = tempMatch[i];
          first_point.push_back(homo_set->kp_first->pt);
          second_point.push_back(homo_set->kp_second->pt);
      }
      Mat Homography_R;
      Mat Homography_T;
      float Homography_score;
      ValidateHomography(first_point,second_point,kfp_ReferenceFrame->Get_IntrinsicParam(),Homography_R,Homography_T,Homography_score);

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

bool NodeHandler::ValidateHomography(vector<Point2f> &arg_kp1, vector<Point2f> &arg_kp2, Mat InstrincParam, Mat& R, Mat& t, float score)
{
  Mat H = findHomography(arg_kp1, arg_kp2, RANSAC);
  cout<<H<<endl;
  vector<Mat> Rs_decomp;
  vector<Mat> Ts_decomp;
  vector<Mat> Normals_decomp;
  int solutions = decomposeHomographyMat(H,InstrincParam,Rs_decomp,Ts_decomp,Normals_decomp);
  cout<<solutions<<endl;
  for (int i = 0; i < 1; i++)
  {
    Mat rvec_decomp;
    cout << "Solution " << i << ":" << endl;
    
    Rodrigues(Rs_decomp[i], rvec_decomp);
    // cout << "rvec from homography decomposition: " <<endl<< Rs_decomp[i] << endl;
    // cout << "tvec from homography decomposition: " <<endl<<  Ts_decomp[i]<<endl;
    Mat projectMatrix = Mat(3,4,CV_64FC1);
    Rs_decomp[i].copyTo(projectMatrix(Rect(0,0,3,3)));
    Ts_decomp[i].copyTo(projectMatrix(Rect(3,0,1,3)));
    Mat InitProjectMatrix = Mat::eye(3,4,CV_64FC1);
    cout << "projection Matrix from homography decomposition: " <<endl<<  projectMatrix<<endl;
    Mat mat_pt1(arg_kp1);
    Mat mat_pt2(arg_kp2);

    Mat dist_coef(1,4,CV_32FC1);//null이나 0값으로 초기화하였다. 
    // Mat mat_Undistorted_pt1;
    // Mat mat_Undistorted_pt2;
    vector<Point2f> Undistorted_pt1;
    Custom_undisortionPoints(arg_kp1,InstrincParam,Undistorted_pt1);//mm단위로 바꿔줍니다.

    // triangulatePoints(InitProjectMatrix,projectMatrix,mat_Undistorted_pt1,mat_Undistorted_pt2,outputMatrix);
    // cout<<"triangulation 후의 결과점 "<<endl;
    // cout<<outputMatrix.t()(Rect(0,0,4,3))<<endl;

    // cout << "plane normal from homography decomposition: " << Normals_decomp[i].t() << endl;
    // double rodrigue_angle = (rvec_decomp.at<double>(0,0)*rvec_decomp.at<double>(0,0)+rvec_decomp.at<double>(1,0)*rvec_decomp.at<double>(1,0)+rvec_decomp.at<double>(2,0)*rvec_decomp.at<double>(2,0));
    // rodrigue_angle = sqrt(rodrigue_angle); 
    // cout<<"실제 각도(degree) : "<<rodrigue_angle*180./3.141592653589793<<endl;
    // cout<<"실제 벡터 : "<<(rvec_decomp/rodrigue_angle).t()<<endl;
    // if(Ts_decomp[i].at<double>(2,0)>=0)
    // {//기존프레임에 대해서 앞으로 이동했을거기 때문에 
    //   cout<<Ts_decomp[i].at<double>(2,0)<<endl;
    // }
    
  }
  exit(0);
}
