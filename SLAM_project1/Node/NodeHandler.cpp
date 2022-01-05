#include "NodeHandler.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "CameraTool.h"
#include <thread>

using namespace std;
using namespace cv;
void CheckRT(int solution_idx,vector<Point2f> current_p2f, vector<Point2f> past_p2f,Mat IntrinsicParam,Mat R, Mat t, int *good_point);
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
          if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)//query -> first
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
      vector<Point2f> current_point;
      vector<Point2f> last_point;

      for( size_t i = 0; i < good_matches.size(); i++ )
      {
          //-- Get the keypoints from the good matches
          Match_Set* homo_set = tempMatch[i];
          current_point.push_back(homo_set->kp_first->pt);//현재프레임
          last_point.push_back(homo_set->kp_second->pt);//나중프레임
      }
      Mat Homography_R;
      Mat Homography_T;
      float Homography_score;
      ValidateHomography(current_point,last_point,kfp_ReferenceFrame->Get_IntrinsicParam(),Homography_R,Homography_T,Homography_score);

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
  Mat H = findHomography(arg_kp2, arg_kp1, RANSAC);
  vector<Mat> Rs_decomp;
  vector<Mat> Ts_decomp;
  vector<Mat> Normals_decomp;
  Mat InstrincParam_64FC1;
  InstrincParam.convertTo(InstrincParam_64FC1,CV_64FC1);
  int solutions = decomposeHomographyMat(H,InstrincParam,Rs_decomp,Ts_decomp,Normals_decomp);//반환해주는 변환은 계사이의 변환이기 때문에 기존의 R,t의 변환이다. 
  //funamental matrix를 구합니다. 

  //https://ros-developer.com/2019/01/01/computing-essential-and-fundamental-matrix-using-opencv-8-points-algorithm-with-c/
  Mat opencv_essential_mat;
  opencv_essential_mat=findEssentialMat(arg_kp2,arg_kp1,InstrincParam);
  Mat outputR;
  Mat outputT;
  Mat Mask;
  recoverPose(opencv_essential_mat,arg_kp2,arg_kp1,InstrincParam,outputR,outputT,Mask);
  Rs_decomp.push_back(outputR);
  Ts_decomp.push_back(outputT);
  int good_points[5] ={0,0,0,0,0};
  thread threads[5];

  for (int i = 0; i < 5; i++)//쓰레드 5개를 돌려서, 각각의 R,t의 가치정보를 얻습니다. 
  {
    threads[i] =  thread(CheckRT,i,arg_kp1,arg_kp2,InstrincParam,Rs_decomp[i],Ts_decomp[i], &good_points[i]); 
  }
  for (int i=0; i<5; i++)
  {
    threads[i].join();
  }
  cout<<"전부 끝났습니다. "<<endl;
  exit(0);
}


void CheckRT(int solution_idx,vector<Point2f> current_p2f, vector<Point2f> past_p2f,Mat IntrinsicParam,Mat R, Mat t, int *good_point)
{
  Mat projectMatrix = Mat(3,4,CV_64FC1);
  R.copyTo(projectMatrix(Rect(0,0,3,3)));
  t.copyTo(projectMatrix(Rect(3,0,1,3)));
  Mat InitProjectMatrix = Mat::eye(3,4,CV_64FC1);

  Mat dist_coef(1,4,CV_64FC1);//null이나 0값으로 초기화하였다.
  vector<Point2f> Undistorted_current_pt;
  Custom_undisortionPoints(current_p2f,IntrinsicParam,Undistorted_current_pt);//mm단위로 바꿔줍니다.
  vector<Point2f> Undistorted_past_pt;
  Custom_undisortionPoints(past_p2f,IntrinsicParam,Undistorted_past_pt);//mm단위로 바꿔줍니다.
  Mat InstrincParam_64FC1;
  IntrinsicParam.convertTo(InstrincParam_64FC1,CV_64FC1);
  vector<Point3f> Output_pt;
  Mat outputMatrix;
  triangulatePoints(InitProjectMatrix,projectMatrix,Undistorted_past_pt,Undistorted_current_pt,outputMatrix);

  vector<Point3d> points;

  for(int point_ind=0; point_ind<outputMatrix.cols; point_ind++)
  {
    Mat x = outputMatrix.col(point_ind);
    x /= x.at<float>(3,0);
    Point3d p (
          x.at<float>(0,0), 
          x.at<float>(1,0), 
          x.at<float>(2,0) 
      );
    points.push_back( p );
  }
  int good_match =0;
  //parallax 변수 : 각 카메라별 좌표 
  Mat PastPt = Mat::zeros(3,1, CV_64F);
  Mat CurrentPt = -R.t()*t;//과거 점 기준
  for(int point_ind=0; point_ind<points.size(); point_ind++)
  {
    Point3d point3d1 = points[point_ind];//3차원점입니다. 과거기준 좌표입니다. 
    Mat point_pose_in_past = Mat(point3d1);
    Mat estimated_point2 = R*(Mat(point3d1)+t);//현재 기준 좌표입니다.~~~~~~~~~~~~~~~~이부분은 이해가 안됨
    Mat PastNormal = point_pose_in_past - PastPt;
    Mat CurrentNormal =point_pose_in_past - estimated_point2;
    float past_dist = norm(PastNormal);
    float current_dist = norm(CurrentNormal);
    float cosParallax = PastNormal.dot(CurrentNormal)/(past_dist*current_dist);
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

    double image1_error = (current_p2f[point_ind].x-projected_pixel_point2.at<double>(0,0))*
                                        (current_p2f[point_ind].x-projected_pixel_point2.at<double>(0,0))
                        + (current_p2f[point_ind].y-projected_pixel_point2.at<double>(1,0))*
                                        (current_p2f[point_ind].y-projected_pixel_point2.at<double>(1,0));

    double image2_error = (past_p2f[point_ind].x-projected_pixel_point1.at<double>(0,0))*
                                        (past_p2f[point_ind].x-projected_pixel_point1.at<double>(0,0))
                        + (past_p2f[point_ind].y-projected_pixel_point1.at<double>(1,0))*
                                        (past_p2f[point_ind].y-projected_pixel_point1.at<double>(1,0));
    if(origin_depth >0)//1. 3차원으로 z값이 0보다 작으면 잘못된 매칭입니다.
    {
      if(image1_error<3 && image2_error<3)
      {
        good_match ++;
      }
    }
  }
  *good_point = good_match;
  cout << "Solution "<<solution_idx<<" answer : " <<*good_point << endl;
}
