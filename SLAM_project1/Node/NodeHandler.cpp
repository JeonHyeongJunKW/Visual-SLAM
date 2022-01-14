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

int NodeHandler::_Get_NumberOfOrbFeature(Mat arg_candidateImage, Mat&des, vector<KeyPoint>& kp)
{
  const static auto& _orb_OrbHandle = ORB::create(1000,1.2,8,31,0,2);
  _orb_OrbHandle->detectAndCompute(arg_candidateImage,noArray(),kp,des);
  return kp.size();
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
  int int_Current_LocalSize = this->_pt_LocalWindowKeyFrames.size();
  if(int_Current_LocalSize ==0){
    return false;}
  else{
    p_lastFrame = this->_pt_LocalWindowKeyFrames.back();//이걸언제 넣을까.
    return true;}
}
bool NodeHandler::AddNewKeyFrame(KeyFrame* p_NewFrame)
{
  if(this->int_CurrentFrameIdx-1>0)
  {
    cout<<"이상한타이밍에 삽입됬습니다."<<endl;
    exit(0);
  }
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
  p_TempFrame->Set_Rt(R,T);
}
void NodeHandler::GetRtParam(KeyFrame* p_TempFrame, double* &R_tparam)
{
  p_TempFrame->Get_Rt(R_tparam);
}
vector<MapPoint*> NodeHandler::Get_localMapPoint()
{
  auto return_vec = _pt_LocalWindowMapPoints;
  return return_vec;
}
void NodeHandler::Add_CandidateKeyFrame(NewKeyFrameSet* matches)
{
  m_sharedlock.lock();
  v_newKeyFrame.push_back(matches);//새롭게 키프레임으로 벡터에 넣어둡니다. 
  m_sharedlock.unlock();
}
void NodeHandler::Match_MapPoint(NewKeyFrameSet *Cu_Re_Matches, Mat camera_global_point)
{
  KeyFrame* currentframe = Cu_Re_Matches->CurrentFrame;
  vector<KeyFrame*> local_keyframe = this->_pt_LocalWindowKeyFrames;
  vector<KeyFrame*> temp_local_keyframe;
  if(local_keyframe.size() ==1 && Cu_Re_Matches->ReferenceFrame->Get_KeyIndex() ==0)
  {//만약에 로컬단위로 유지하던 키프레임의 수가 0이라면 그냥 처음 레피런스 프레임이랑 유지하던 맵포인트를 넣습니다.  
    Mat current_descriptors = Cu_Re_Matches->descriptor;//현재 프레임에 대한 descriptor들
    for(int j=0; j<Cu_Re_Matches->CurrentGoodPoint2D.size();j++)
    {//맵포인트 생성 및 새로운 맵포인트를 만듭니다. 
      MapPoint* new_mapPoint =new MapPoint();
      new_mapPoint->int_Node = _int_MapPointIdx;//새로운 맵포인트에 인덱스를 부여합니다. 
      new_mapPoint->make_count = 0;
      new_mapPoint->view_count = 2;
      current_descriptors(Rect(0,j,32,1)).copyTo(new_mapPoint->mat_Orbdescriptor);//기존 레퍼런스 프레임과 사용하던 맵포인트를 등록합니다. 
      new_mapPoint->max_d = 10;//애매하다.
      new_mapPoint->min_d = 1;//애매하다.
      // Mat local_point3f(Cu_Re_Matches->CurrentGoodPoint3D[j]);
      Mat local_point3d = (Mat_<double>(4,1)<<Cu_Re_Matches->CurrentGoodPoint3D[j].x,
                                            Cu_Re_Matches->CurrentGoodPoint3D[j].y,
                                            Cu_Re_Matches->CurrentGoodPoint3D[j].z,
                                            1);
      // local_point3d.convertTo(local_point3d,CV_32FC1);
      Mat global_point3f =camera_global_point*local_point3d;
      global_point3f /= global_point3f.at<double>(3,0);
      new_mapPoint->p3d_coordinate = Point3f(global_point3f.at<double>(0,0),
                                            global_point3f.at<double>(1,0),
                                            global_point3f.at<double>(2,0));//글로벌 3차원좌표로 바꿔서 저장합니다. 
      new_mapPoint->scale = 1;//애매하다.
      new_mapPoint->pixel_match[Cu_Re_Matches->CurrentFrame->Get_KeyIndex()] = Cu_Re_Matches->CurrentGoodPoint2D[j];
      new_mapPoint->pixel_match[Cu_Re_Matches->ReferenceFrame->Get_KeyIndex()] = Cu_Re_Matches->ReferenceGoodPoint2D[j];
      
      if(Cu_Re_Matches->ReferenceFrame->Get_KeyIndex()!=0)
      {
        cout<<"이상합니다."<<endl;
        exit(0);
      }
      Cu_Re_Matches->CurrentFrame->Add_MapPoint(new_mapPoint);//현재 프레임에 맵포인트를 추가합니다.
      Cu_Re_Matches->ReferenceFrame->Add_MapPoint(new_mapPoint);//래퍼런스 프레임에 맵포인트를 추가합니다.
      _int_MapPointIdx++;
    }
    
    temp_local_keyframe.push_back(Cu_Re_Matches->ReferenceFrame);
    temp_local_keyframe.push_back(Cu_Re_Matches->CurrentFrame);
  }
  else
  {//현재 로컬프레임이랑 매칭을 합니다. 
    for(int i =0; i<this->_pt_LocalWindowKeyFrames.size(); i++)
    {//하나씩 비교합니다.
      
      KeyFrame* candidate_connected_frame = _pt_LocalWindowKeyFrames[i];//로컬프레임의 i번째 프레임을 얻어옵니다.
      vector<vector<DMatch>> matches;//매치를 저장할 변수입니다.
      _match_OrbMatchHandle->knnMatch(currentframe->Get_Descriptor(),
                                              candidate_connected_frame->Get_Descriptor(),
                                              matches,
                                              2);//쿼리 디스크립터(현재 키프레임에서 나온 맵포인트)와 맞는 로컬 디스크립터(로컬 맵포인트)를 
                                              //찾습니다.
      const float ratio_thresh = 0.65f;//비교 임계값
      vector<DMatch> good_matches;
      for (size_t match_ind = 0; match_ind< matches.size(); match_ind++)
      {
        if (matches[match_ind][0].distance < ratio_thresh * matches[match_ind][1].distance)
        {
          good_matches.push_back(matches[match_ind][0]);//매칭이 잘된 경우
        }
      }
      
      //매칭이 잘된경우에는 일단 triangulation을 하여 비교한다. 
      // cout<<"-----------------"<<endl;
      // cout<<"매칭 수 :"<<good_matches.size()<<endl;
      int good_point_thres = good_matches.size()-15;
      if(good_point_thres<0)//매칭이 잘안된경우에는 skip한다. 
      {
        continue;
      }
      //매칭이 좋은점들을 좌표 벡터로 만든다. 
      //좌표벡터로 triangulation을 한다.(서로간에 R,t 관계로 확인) 
      //깊이를 확인하고 매칭결과를 본다.
      //스케일도 비교한다. 
      
      vector<Point2d> current_point;
      vector<Point2d> candidate_point;

      int cu_max_size = currentframe->Get_keyPoint().size();
      int ca_max_size = candidate_connected_frame->Get_keyPoint().size();
      vector<KeyPoint> cur_keypoints = currentframe->Get_keyPoint();
      vector<KeyPoint> cand_keypoints = candidate_connected_frame->Get_keyPoint();
      for( int match_ind = 0; match_ind < good_matches.size(); match_ind++ )
      {
        int curr_Idx = good_matches[match_ind].queryIdx;
        KeyPoint cur_keypoint = cur_keypoints[curr_Idx];
        Point2f cur_point_2f = cur_keypoint.pt;
        int cand_Idx = good_matches[match_ind].trainIdx;
        KeyPoint cand_keypoint = cand_keypoints[cand_Idx];
        Point2f cand_point_2f = cand_keypoint.pt;

        current_point.push_back(Point2d(double(cur_point_2f.x),double(cur_point_2f.y)));//현재프레임
        candidate_point.push_back(Point2d(double(cand_point_2f.x),double(cand_point_2f.y)));//나중프레임
      }

      Mat last_global_pose = candidate_connected_frame->global_pose;//후보 키프레임의 전역포즈를 얻어옵니다.
      
      Mat projection_matrix_4_3 =(last_global_pose.inv()*camera_global_point).inv()(Rect(0,0,4,3));//상대변환에 해당하는 R,t를 구하는부분입니다.
      

      projection_matrix_4_3.convertTo(projection_matrix_4_3,CV_64FC1);
      Mat InitProjectMatrix = Mat::eye(3,4,CV_64FC1);
      Mat dist_coef= Mat::zeros(1,4,CV_64FC1);//null값으로 초기화하였다.
      Mat InstricParam  =currentframe->Get_IntrinsicParam();
      vector<Point2d> Undistorted_current_pt;
      Custom_undisortionPoints(current_point,InstricParam,Undistorted_current_pt);//mm단위로 바꿔줍니다.
      vector<Point2d> Undistorted_candiate_pt;
      Custom_undisortionPoints(candidate_point,InstricParam,Undistorted_candiate_pt);//mm단위로 바꿔줍니다.

      Mat InstrincParam_64FC1;
      InstricParam.convertTo(InstrincParam_64FC1,CV_64FC1);
      vector<Point3f> Output_pt;
      Mat outputMatrix;
      triangulatePoints(InitProjectMatrix,projection_matrix_4_3,Undistorted_candiate_pt,Undistorted_current_pt,outputMatrix);
      vector<Point3d> points;//triangulation을 통한 3차원점
      for(int point_ind=0; point_ind<outputMatrix.cols; point_ind++)
      {
        Mat x = outputMatrix.col(point_ind);
        
        x /= x.at<double>(3,0);
        Point3d p (
              x.at<double>(0,0), 
              x.at<double>(1,0), 
              x.at<double>(2,0) 
          );
        points.push_back(p);
      }
      int good_match =0;
      //parallax 변수 : 각 카메라별 좌표 
      Mat PastPt = Mat::zeros(3,1, CV_64FC1);
      
      Mat R = projection_matrix_4_3(Rect(0,0,3,3));
      
      Mat t = projection_matrix_4_3(Rect(3,0,1,3));
      Mat CurrentPt = -R.t()*t;//과거 점 기준
      map<int, MapPoint*> candidate_map_point = candidate_connected_frame->Get_MapPoint();//Get 맵포인트함수를 통해서 후보 프레임에 대한 맵포인트를 가져옵니다. 
      bool is_good = false;
      int matched_point =0;
      for(int point_ind=0; point_ind<points.size(); point_ind++)//현재 프레임에서 얻은 3차원점들에 대해서 
      {
        // cout<<"후보 "<<points.size() <<" : "<<point_ind<<endl;
        Point3d point3d1 = points[point_ind];//3차원점입니다. 과거기준 좌표입니다. 
        Mat point_pose_in_past = Mat(point3d1);
        Mat estimated_point2 = R*(Mat(point3d1)+t);//현재 기준 좌표
        Mat PastNormal = point_pose_in_past - PastPt;
        
        Mat CurrentNormal =point_pose_in_past - estimated_point2;
        double past_dist = norm(PastNormal);
        double current_dist = norm(CurrentNormal);
        double cosParallax = PastNormal.dot(CurrentNormal)/(past_dist*current_dist+0.0000000000001);
        if(cosParallax> 0.9999)
        {
          continue;
        }
        Point3d point3d2 (//현재기준 좌표입니다.
              estimated_point2.at<double>(0,0), 
              estimated_point2.at<double>(1,0), 
              estimated_point2.at<double>(2,0) 
          );
        double origin_depth = estimated_point2.at<double>(2,0);

        point3d1 /= point3d1.z; //정규화를 합니다.
        point3d2 /= point3d2.z; //정규화를 합니다. 
        Mat projected_pixel_point1 = InstrincParam_64FC1*Mat(point3d1);//카메라 파라미터를 곱해서 원래 과거 픽셀좌표로 바꿉니다. 
        Mat projected_pixel_point2 = InstrincParam_64FC1*Mat(point3d2);//카메라 파라미터를 곱해서 원래 현재 픽셀좌표로 바꿉니다. 
        double image1_error = (current_point[point_ind].x-projected_pixel_point2.at<double>(0,0))*
                                            (current_point[point_ind].x-projected_pixel_point2.at<double>(0,0))
                            + (current_point[point_ind].y-projected_pixel_point2.at<double>(1,0))*
                                            (current_point[point_ind].y-projected_pixel_point2.at<double>(1,0));

        double image2_error = (candidate_point[point_ind].x-projected_pixel_point1.at<double>(0,0))*
                                            (candidate_point[point_ind].x-projected_pixel_point1.at<double>(0,0))
                            + (candidate_point[point_ind].y-projected_pixel_point1.at<double>(1,0))*
                                            (candidate_point[point_ind].y-projected_pixel_point1.at<double>(1,0));                                
        if(origin_depth <0)//1. 3차원으로 z값이 0보다 작으면 잘못된 매칭입니다.
        {
          continue;
        }
        if(image1_error >15 || image2_error>15)//reprojection error가 적당한 임계값을 만족하는지 확인합니다. 
        {
          continue;
        }
        matched_point++;
        //스케일 에러도 찾아야하는데 귀찮다.. 미루자.
        //이미 레퍼런스 프레임에서 유지하고 있는 맵포인트 정보도 얻는다. 픽셀이랑 3차원좌표등에서 매우 유사한 점을 찾는다. 
        //해당 맵포인트를 업데이트하거나 새로만든다. (맵포인트에서 이미 해당 프레임을 유지하고 있는 지를 확인해서 등록한다.)
        int min_range =10000;
        int min_index = 0;
        // cout<<"맵포인트 이웃삽입전"<<endl; 
        auto p_candidate_mappoint = candidate_map_point.begin();
        for(int k=0; k<candidate_map_point.size(); k++)//그래서 뭐햇냐면 해당 후보프레임이 가지고 잇는 맵포인트 내놓으라고 햇어 
        {
          // cout<<k<< " : "<<endl;
          map<int,Point2d> cand_match = p_candidate_mappoint->second->pixel_match;
          
          int current_frame_idx = candidate_connected_frame->Get_KeyIndex();
          if(cand_match.count(current_frame_idx) == 0)
          {
            cout<<"맞는 키를 못찾았습니다."<<endl;//0번프레임이 가지고 있는 맵포인트를 가져오라고했는데, 해당맵포인트가 정작 비교프레임에 대한 2D좌표가 없어
            for(auto cand_it= cand_match.begin(); cand_it != cand_match.end();cand_it++)
            {
              cout<<cand_it->first<<" "<<cand_it->second<<endl;
            }
            cout<<cand_match.size()<<" "<<current_frame_idx<<" "<<currentframe->Get_KeyIndex()<<endl;
            exit(0);
          }
          Point2d temp_match_point = cand_match[current_frame_idx];//현재 후보 프레임의 맵포인트를 현재 
          //프레임과 맞는걸로 가져옵니다.
          
          if(norm(temp_match_point-candidate_point[point_ind])<min_range)//현재 점과의 거리를 비교합니다. 
          {
            min_range = norm(temp_match_point-candidate_point[point_ind]);
            min_index = p_candidate_mappoint->first;//키값을 저장합니다. 
          }
          p_candidate_mappoint++;
        }
        // cout<<"맵포인트 이웃삽입후"<<endl; 
        is_good = true;
        if(min_range<3)//가깝다면 해당점을 추가합니다.
        {
          MapPoint* old_mapPoint =candidate_map_point[min_index];
          old_mapPoint->view_count +=1;
          // current_descriptors(Rect(0,i,32,1)).copyTo(new_mapPoint->mat_Orbdescriptor);
          old_mapPoint->pixel_match[Cu_Re_Matches->CurrentFrame->Get_KeyIndex()] = currentframe->Get_keyPoint()[good_matches[point_ind].queryIdx].pt;
          //현재 키프레임에 해당 맵포인트를 등록합니다. 
          currentframe->Add_MapPoint(old_mapPoint);
        }
        else//아니라면 새로운 맵포인트를 등록합니다. 
        {
          if(min_range ==10000)//너무 차이가 많이 나는 안좋은 점입니다. 그럼 버립니다. 
          {
            cout<<"이럴리가 없는데, "<<endl;
            cout<<"후보 프레임의 인덱스 : "<<candidate_connected_frame->Get_KeyIndex()<<endl;
            cout<<"현재 프레임의 인덱스 : "<<currentframe->Get_KeyIndex()<<endl;
            cout<<"후보 맵프레임의 맵포인트 사이즈 : "<<candidate_map_point.size()<<endl;
            exit(0);
          }
          else
          {
            MapPoint* new_mapPoint =new MapPoint();
            new_mapPoint->int_Node = _int_MapPointIdx;//새로운 맵포인트에 인덱스를 부여합니다. 
            // if(_int_MapPointIdx == 483)
            // {
            //   cout<<"최종입력 맵포인트"<<_int_MapPointIdx-1<<endl;
            //   exit(0);
            // }
            
            new_mapPoint->make_count = 0;
            new_mapPoint->view_count = 2;
            (currentframe->Get_Descriptor())(Rect(0,good_matches[point_ind].queryIdx,32,1)).copyTo(new_mapPoint->mat_Orbdescriptor);//기존 레퍼런스 프레임과 사용하던 맵포인트를 등록합니다. 
            new_mapPoint->max_d = 10;//애매하다.
            new_mapPoint->min_d = 1;//애매하다.
            // Mat local_point3f(Cu_Re_Matches->CurrentGoodPoint3D[j]);
            
            Mat local_point3f = (Mat_<double>(4,1)<<point3d2.x,
                                                point3d2.y,
                                                point3d2.z,
                                                1);
            Mat global_point3f = camera_global_point*local_point3f;//점에 대한 변환으로 바꿔서 global 좌표로 구합니다. 
            global_point3f /= global_point3f.at<double>(3,0);
            new_mapPoint->p3d_coordinate = Point3f(global_point3f.at<double>(0,0),
                                            global_point3f.at<double>(1,0),
                                            global_point3f.at<double>(2,0));//글로벌 3차원좌표로 바꿔서 저장합니다. 
            new_mapPoint->scale = 1;//애매하다.
            new_mapPoint->pixel_match[Cu_Re_Matches->CurrentFrame->Get_KeyIndex()] = currentframe->Get_keyPoint()[good_matches[point_ind].queryIdx].pt;
            new_mapPoint->pixel_match[candidate_connected_frame->Get_KeyIndex()] = candidate_connected_frame->Get_keyPoint()[good_matches[point_ind].trainIdx].pt;
            currentframe->Add_MapPoint(new_mapPoint);//현재 프레임에 맵포인트를 추가합니다.
            
            candidate_connected_frame->Add_MapPoint(new_mapPoint);//래퍼런스 프레임에 맵포인트를 추가합니다.
            _int_MapPointIdx++;
          }
        }
      }
      if (matched_point !=0)
      {
        temp_local_keyframe.push_back(candidate_connected_frame);
        // cout<<"local frame 삽입이 끝났습니다."<<endl;
      }
      
    }
    // cout<<"마지막으로 현재프레임을 삽입합니다."<<endl;
    temp_local_keyframe.push_back(Cu_Re_Matches->CurrentFrame);
    
  }
  this->_pt_LocalWindowKeyFrames.clear();
  this->_pt_LocalWindowKeyFrames = temp_local_keyframe;//로컬키프레임을 업데이트합니다. 
  // std::cout<<"현재 로컬 맵프레임 사이즈 : "<<_pt_LocalWindowKeyFrames.size()<<endl;
  // cout<<"로컬윈도우를 업데이트합니다."<<endl;
}