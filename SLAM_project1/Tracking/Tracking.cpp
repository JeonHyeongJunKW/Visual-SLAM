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
          cout<<"kfp last frame : "<<kfp_LastFrame<<endl;
          nodehandler.SetImageFeature(kfp_TempFrame,candidate_image);//현재 프레임 정보를 등록합니다.
          if(!b_IsHaveLocalFrame)//초기 키프레임이 없다면 초기프레임을 등록합니다. 
          {
            nodehandler.AddNewKeyFrame(kfp_TempFrame);//노드 핸들러에 새프레임을 등록합니다. 
            continue;
          }
          //이미 프레임이 있다면 마지막 프레임으로 R,t를 구합니다. 
          Mat Initial_R;
          Mat Initial_T;
          NewKeyFrameSet* Reference_matches = new NewKeyFrameSet();//마지막점과의 매칭 관계를 저장한다.

          GetInitialRt(nodehandler,
                      kfp_TempFrame,
                      kfp_LastFrame,
                      Initial_R,
                      Initial_T, 
                      Reference_matches);

          nodehandler.SetRt(kfp_TempFrame,Initial_R,Initial_T);

          vector<Match_Set> mappoint_match;
          bool IsZeroPoint = track_localMap(nodehandler,
                                          kfp_TempFrame,
                                          mappoint_match);//현재 프레임에 대해서 local mapping을 합니다. 

          if(IsZeroPoint)
          {
            nodehandler.Add_CandidateKeyFrame(Reference_matches);//노드 핸들러에 새프레임을 등록합니다.
          }
          else
          {
            bool IsGoodKeyFrame = decide_newkeyframe(nodehandler,kfp_TempFrame);//괜찮은 프레임인지 검사합니다.
            nodehandler.AddNewKeyFrame(kfp_TempFrame);//노드 핸들러에 새프레임을 등록합니다.
          }
        }
        waitKey(1);
    }
    
}

bool GetInitialRt(NodeHandler &nodehandler, //노드(키프레임, 맵포인트)를 저장및 처리하기위한 툴입니다.
                  KeyFrame* kfp_NewFrame, //현재 키프레임입니다.
                  KeyFrame* kfp_LastFrame, //마지막에 넣은 키프레임입니다.
                  Mat &R, //반환되는 초기 R값입니다.
                  Mat &t, //반환되는 초기 t값입니다.
                  NewKeyFrameSet *Cu_Re_Matches) 
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
    }
    else
    {
      mode = FUND;
      inlier_set = f_inlier;
    }
    
    for(int j=0; j<inlier_set.size(); j++)
    {
      inliner_current_point.push_back(current_point[inlier_set[j]]);
      inliner_reference_point.push_back(last_point[inlier_set[j]]);
    }
    vector<int> final_good_point_ind;//inliner_current_point안에서도 좋은 매칭을 가지는 점의 인덱스들입니다.
    vector<Point3d> current_good_point_3d;//inliner_current_point안에서도 좋은 매칭을 가지는 점의 3차원좌표입니다.
    if(mode ==HOMO)
    {
      ValidateHomographyRt(inliner_current_point,
                          inliner_reference_point,
                          lastFrame->Get_IntrinsicParam(),
                          return_R,
                          return_T,
                          final_good_point_ind,
                          current_good_point_3d);
    }
    else
    {
      
      ValidateFundamentalRt(inliner_current_point,
                          inliner_reference_point,
                          lastFrame->Get_IntrinsicParam(),
                          return_R,
                          return_T,
                          final_good_point_ind,
                          current_good_point_3d);
    }
    
    R= return_R;
    t =return_T;
    
    Cu_Re_Matches->CurrentFrame = kfp_NewFrame;
    Cu_Re_Matches->ReferenceFrame = kfp_LastFrame;
    Cu_Re_Matches->CurrentGoodPoint3D = current_good_point_3d;
    vector<Point2f> CurretGoodPoint2D;
    vector<Point2f> ReferenceGoodPoint2D;
    vector<Mat> descriptor;
    Mat all_descriptor =kfp_NewFrame->Get_Descriptor();
    for(int i=0; i<final_good_point_ind.size(); i++)
    {
      int good_match_ind = inlier_set[final_good_point_ind[i]];
      CurretGoodPoint2D.push_back(current_point[good_match_ind]);
      ReferenceGoodPoint2D.push_back(last_point[good_match_ind]);
      descriptor.push_back(all_descriptor(Rect(0,good_match_ind,32,1)));
    }
    Cu_Re_Matches->CurrentGoodPoint2D = CurretGoodPoint2D;
    Cu_Re_Matches->ReferenceGoodPoint2D = ReferenceGoodPoint2D;
    Cu_Re_Matches->descriptor = descriptor;
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