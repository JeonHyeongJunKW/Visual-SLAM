#include "Tracking.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Node/NodeHandler.h"
#include "../Node/CameraTool.h"
#include <thread>
using namespace std;
using namespace cv;

void tracking_thread(NodeHandler& nodehandler)
{   
    vector<cv::String> file_names;
    glob("/media/jeon/T7/Kitti dataset/data_odometry_gray/dataset/sequences/00/image_0/*.png", file_names, false);

    for(int camera_ind=0; camera_ind<file_names.size(); camera_ind++)
    { 
        Mat candidate_image = imread(file_names[camera_ind],IMREAD_GRAYSCALE);//전체 파일들을 불러옵니다.
        imshow("camera image",candidate_image);
        bool b_IsNiceTime = nodehandler.IsNiceTime();//키프레임 넣은 후에 적당한 시간이 지났는지 검사합니다. 
        if(b_IsNiceTime)
        {
            KeyFrame* kfp_TempFrame = new KeyFrame(nodehandler.int_CurrentFrameIdx-1, 
                                                    nodehandler.mat_InstrisicParam);//해당 키프레임 포인터 가져오고
            KeyFrame* kfp_LastFrame;
            bool b_IsHaveLocalFrame = nodehandler.GetLastFrame(kfp_LastFrame);//마지막으로 키프레임으로 등록한 프레임을 가져옵니다. 
            nodehandler.SetImageFeature(kfp_TempFrame,candidate_image);//현재 프레임 정보를 등록합니다.
            if(!b_IsHaveLocalFrame)//초기 키프레임이 없다면 초기프레임을 등록합니다. 
            {
              
              nodehandler.AddNewKeyFrame(kfp_TempFrame);//노드 핸들러에 새프레임을 등록합니다. 
              continue;
            }
            //이미 프레임이 있다면 마지막 프레임으로 R,t를 구합니다. 
            Mat Initial_R;
            Mat Initial_T;
            
            GetInitialRt(nodehandler,kfp_TempFrame,kfp_LastFrame,Initial_R,Initial_T);
            nodehandler.SetRt(kfp_TempFrame,Initial_R,Initial_T);

            vector<Match_Set> mappoint_match;
            bool IsZeroPoint = track_localMap(nodehandler,kfp_TempFrame,mappoint_match);

            if(IsZeroPoint)
            {
              cout<<"점이 없어요."<<endl;
              nodehandler.AddNewKeyFrame(kfp_TempFrame);//노드 핸들러에 새프레임을 등록합니다.
            }
            else
            {
              cout<<"점이 있어요."<<endl;
              bool IsGoodKeyFrame = decide_newkeyframe(nodehandler,kfp_TempFrame);
              nodehandler.AddNewKeyFrame(kfp_TempFrame);//노드 핸들러에 새프레임을 등록합니다.
            }
            
            
            exit(0);
        }
        waitKey(1);
    }
}

bool GetInitialRt(NodeHandler &nodehandler, KeyFrame* kfp_NewFrame,KeyFrame* kfp_LastFrame, Mat &R, Mat &t) 
{
    KeyFrame* lastFrame = kfp_LastFrame;
    vector<vector<DMatch>> matches;//매치를 저장할 변수입니다.
    nodehandler._match_OrbMatchHandle->knnMatch(kfp_NewFrame->Get_Descriptor(),kfp_LastFrame->Get_Descriptor(),matches,2);//쿼리 디스크립터를 찾습니다. 
    //initial pose estimation from last frame
    vector<DMatch> good_matches;//좋은 매치만 저장됩니다.
    const float ratio_thresh = 0.65f;
    
    for (size_t i = 0; i < matches.size(); i++){
      if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)//query -> first
      {
        good_matches.push_back(matches[i][0]);
      }
    }
    if(good_matches.size()<5)//현재 로컬프레임과의 매칭이 잘되었는지 확인합니다. 
    {
      cout<<"relocalization start"<<endl;
      exit(0);
    }

    vector<Point2f> current_point;
    vector<Point2f> last_point;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
      current_point.push_back(kfp_NewFrame->Get_keyPoint()[good_matches[i].queryIdx].pt);//현재프레임
      last_point.push_back(lastFrame->Get_keyPoint()[good_matches[i].trainIdx].pt);//나중프레임
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
    const int HOMO =0;
    const int FUND =1;

    if(sh/(sh+sf) >0.45)
    {
      mode = HOMO;
      inlier_set = h_inlier;
      // cout<<"model is homography"<<endl; 
      // cout<<"인라이어의 수 : "<<inlier_set.size()<<endl;
    }
    else
    {
      mode = FUND;
      inlier_set = f_inlier;
      // cout<<"model is Fundamental"<<endl;
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
    R= return_R;
    t =return_T;
}
bool track_localMap(NodeHandler &nodehandler, KeyFrame* kfp_NewFrame,vector<Match_Set> &localmap)
{
  vector<MapPoint*>  local_mappoint = nodehandler.Get_localMapPoint();
  int size_mappoint = local_mappoint.size();
  if(size_mappoint ==0)//트랙킹하는 맵포인트가 없습니다.
  {
    return true;
  }
  else
  {
    return false;
  }
}
bool decide_newkeyframe(NodeHandler &nodehandler, KeyFrame* kfp_NewFrame)
{
  return true;
}