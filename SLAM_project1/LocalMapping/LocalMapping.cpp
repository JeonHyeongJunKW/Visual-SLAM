#include "LocalMapping.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Node/NodeHandler.h"
#include "../Node/CameraTool.h"
#include "math.h"
using namespace std;
using namespace cv;

void LocalMapping(NodeHandler &nodehandler)
{
    //--지도상에 추정치를 그리는부분 초기화--begin
    char filename[] = "/media/jeon/T7/Kitti dataset/data_odometry_poses/dataset/poses/00.txt";
    //--지도상에 추정치를 그리는부분 초기화--end

    Mat origin_pose = load_origin_pose(filename);//초기카메라 위치를 얻어옵니다. 
    Mat estimated_pose = origin_pose;//추정될카메라의 누적된 위치를 기록하는 행렬입니다.
    while(true)
    {
        if(nodehandler.v_newKeyFrame.size()>0)//키프레임 후보가 추가적으로 있는지, 벡터안에 갯수로 확인합니다.
        {
            //--실제 업데이트하는 부분
            
            nodehandler.m_sharedlock.lock();//뮤텍스락으로 키프레임에 대한 노드핸들링권한을 가져옵니다.
            NewKeyFrameSet* newFrameSet =nodehandler.v_newKeyFrame[0];//새로운 프레임 매칭에 대한 정보를 얻어옵니다.
            float* R_t_scale;//13개의 파라미터가 순서대로 나옵니다.
            newFrameSet->CurrentFrame->Get_Rt(R_t_scale);//현재 키프레임의 R,t,scale정보의 초기값을 얻어옵니다.

            Mat Extrinsic_matrix = //카메라 외부 행렬을 가져옵니다.
            (Mat_<float>(4,4)    
            <<R_t_scale[0],R_t_scale[1],R_t_scale[2],R_t_scale[3],
              R_t_scale[4],R_t_scale[5],R_t_scale[6],R_t_scale[7],
              R_t_scale[8],R_t_scale[9],R_t_scale[10],R_t_scale[11],
                        0.,          0.,           0.,R_t_scale[12]);

            estimated_pose = estimated_pose*Extrinsic_matrix.inv(); //카메라 외부 행렬의 역행렬로 초기 카메라 위치계에 대한 변환으로 업데이트합니다. 
            nodehandler.AddNewKeyFrame(nodehandler.v_newKeyFrame[0]->CurrentFrame); //새로운 키프레임으로 등록합니다.
            nodehandler.v_newKeyFrame.erase(nodehandler.v_newKeyFrame.begin());//현재 키프레임후보에서 방금 처리한 키프레임을 없앱니다. 
            nodehandler.estimated_pose.push_back(estimated_pose);//(그리는용도) 현재 추정된 위치를 그립니다. 
            //--실제 업데이트하는 부분 끝
            
            
            nodehandler.m_sharedlock.unlock();//뮤텍스락으로 노드핸들링권한을 반납하여, tracking node가 일하게합니다.
        }
    }
}

Mat load_origin_pose(const char* filename)
{//ground truth의 초기위치를 가져오는 함수
  ifstream poseFile;//초기자세를 읽어올 파일스트림
  poseFile.open(filename);//파일스트림으로 초기자세가 담겨있는 파일을 얻어옵니다.
  Mat return_origin;//초기자세를 읽어서 저장합니다. 

  if(poseFile.is_open())//파일이 열려있는지 확인합니다.
  {
      float p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12;//파일스트림으로 읽어올 초기자세 행렬의 원소입니다.
      string mat_pose;//초기자세를 txt 파일에서 한줄씩 읽어서 저장할 string객체입니다.
      getline(poseFile,mat_pose);//string 타입으로 초기자세를 읽어옵니다.
      sscanf(mat_pose.c_str(),"%e %e %e %e %e %e %e %e %e %e %e %e",//읽어온 초기자세를 다시 숫자로 받기위해서 e타입으로 읽습니다.
            &p1,&p2,&p3,&p4,&p5,&p6,&p7,&p8,&p9,&p10,&p11,&p12);
      return_origin=(Mat_<float>(4,4) <<p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,0,0,0,1);//초기자세를 Mat의 형태로 얻습니다.
  }
  return return_origin;
}