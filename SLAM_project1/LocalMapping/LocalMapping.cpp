#include "LocalMapping.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Node/NodeHandler.h"
#include "../Node/CameraTool.h"
#include "math.h"
using namespace std;
using namespace cv;
void Optimize_localmap( vector<map_point_kch> &MapPointSet, map<int,key_frame_kch> &keyFrameSet)
{
    //찬혁이가 해야하는 부분
    //로컬 키프레임 내에 존재하는 맵포인트와 키프레임입니다. 맵포인트의 관측을 통해서 최적화 식을 만들어야합니다. 
}

void LocalMapping(NodeHandler &nodehandler)
{
    char filename[] = "/media/jeon/T7/Kitti dataset/data_odometry_poses/dataset/poses/00.txt";

    Mat origin_pose = load_origin_pose(filename);//초기카메라 위치를 얻어옵니다. 
    Mat estimated_pose = origin_pose;//추정될카메라의 누적된 위치를 기록하는 행렬입니다.
    bool is_new_start = true;
    while(true)
    {//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!읽어주세요.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        /* 이 아래는 로컬 맵핑부분입니다. 다음과같은 순서로 이루어집니다. 
        # < 1 step > tracking 부분에서 전달해준, 초기 R,t를 이용하여, 전역 R,t를 업데이트합니다. 현재 키프레임에도 전역 R,t를 등록합니다. 

        # < 2 step > 현재 키프레임void Get_OptimizeSet(vector<KeyFrame*> past_localframe, vector<map_point_kch> &InitialMapPointSet, map<int,key_frame_kch> &InitialkeyFrameSet);
void Set_OptimizeSet(vector<KeyFrame*> &past_localframe, vector<map_point_kch> InitialMapPointSet, map<int,key_frame_kch> InitialkeyFrameSet);합니다. (로컬 키프레임들은 vector<KeyFrame*> nodehandler._pt_LocalWindowKeyFrames
                    안에 포인터의 형태로 저장되어있음)

        # < 3 step > < 2 step > 전에 유지하던 로컬 키프레임들(clone_past_localframe)의 맵포인트에 대해서 생성이후에 지난 시간(make_count)를             업데이트하고,   
                    이후에 보였는지(view_count)를 기반으로 하여 맵포인트를 삭제합니다. 논문에서는 처음삽입이후에 3번이상 보였는지를 기반으로 삭제하는듯 합니다. 추가적으로 필요한 변수가 있는지 알려주세요. 참고로 각 키프레임에서 _pmappoint_OwnedMapPoint라는 map<int, MapPoint*> 에 맵포인트들의 고유 인덱스가 key로, 맵포인트들의 포인터가 값으로 저장되어있어서, 맵포인트를 제거한다면 이 map 에서도 제거해줘야합니다. 

         # < 4 step > 컬링이 완료된 로컬 맵포인트들과 현재 키프레임에 대해서 관측된 맵포인트들과 키프레임들사이의 R,t관계를 구하고, 최적화합니다. 화이팅
        */
       
        nodehandler.m_sharedlock.lock();//뮤텍스락으로 키프레임에 대한 노드핸들링권한을 가져옵니다.
        if(nodehandler.v_newKeyFrame.size()>0)//키프레임 후보가 추가적으로 있는지, 벡터안에 갯수로 확인합니다.
        {
            //# < 1 step > tracking 부분에서 전달해준, 초기 R,t를 이용하여, 전역 R,t를 업데이트합니다. 
            NewKeyFrameSet* newFrameSet =nodehandler.v_newKeyFrame[0];//새로운 프레임 매칭(현재 키프레임과 이전 키프레임)에 
                                                                    //대한 정보를 얻어옵니다.
            if(is_new_start)//키프레임 삽입과정인지 확인합니다.(초기화 플래그를 확인합니다.)
            {
                is_new_start = false;//초기화 플래그를 false로 바꿉니다. 
                Mat copyed_ref_pose(4,4,CV_64FC1);//0번째(초기) 프레임에 대한 포즈를 저장하는 행렬입니다.
                estimated_pose.copyTo(copyed_ref_pose);//초기 포즈를 0번째 프레임에 대한 카메라 포즈로 복사합니다. 
                
                newFrameSet->ReferenceFrame->global_pose = copyed_ref_pose;//0번째 프레임에 대한 카메라 포즈를 저장합니다. 
            }
            else
            {
                nodehandler._pt_LocalWindowKeyFrames.back()->global_pose.copyTo(estimated_pose);
            }
            double* R_t_scale;//13개의 외부 파라미터를 저장할 변수입니다. 
            newFrameSet->CurrentFrame->Get_Rt(R_t_scale);//현재 키프레임의 R,t,scale정보의 초기값을 얻어옵니다.(이전 프레임에 대한)
            Mat Extrinsic_matrix = R_t_scale_2_Mat(R_t_scale);//위에 함수에서 얻은 R,t,scale정보의 초기값을 행렬로 바꿔서 저장합니다.
            // cout<<"현재 R,t ("<<newFrameSet->CurrentFrame->Get_KeyIndex()<<")"<<endl;//현재 키프레임의 인덱스를 표시합니다. 
            
            estimated_pose = estimated_pose*Extrinsic_matrix.inv(); //카메라 외부 행렬의 역행렬로 global계에 대한 변환으로 현재 카메라 포즈를 업데이트합니다. 

            Mat copyed_cur_pose(4,4,CV_64FC1);//global계에 대한 현재 카메라 포즈를 복사해서 저장하는 행렬입니다.
            estimated_pose.copyTo(copyed_cur_pose);//현재 카메라 포즈를 복사해서 위의 변수에 저장합니다. 
            newFrameSet->CurrentFrame->global_pose = copyed_cur_pose;//global 계에 대한 현재 카메라 포즈를 현재 키프레임에 저장합니다. 


            //# < 2 step > 현재 키프레임을 로컬 키프레임에 추가하면서 맵포인트를 초기 연결을 시켜줍니다. 이때, 맵포인트 점들의 투영간에 정보를 
            //        활용하여 매칭합니다. (로컬 키프레임들은 vector<KeyFrame*> nodehandler._pt_LocalWindowKeyFrames
            //        안에 포인터의 형태로 저장되어있음)

            Mat copyed_pose(4,4,CV_64FC1);//global계에 대한 현재 카메라 포즈를 복사해서 저장하는 행렬입니다. 로컬맵포인트 과정간에 사용됩니다.
            estimated_pose.copyTo(copyed_pose);//현재 카메라 포즈를 복사해서 위의 변수에 저장합니다.
            vector<KeyFrame*> clone_past_localframe = nodehandler._pt_LocalWindowKeyFrames;//임시로 로컬키프레임들을 저장합니다. 
            nodehandler.Match_MapPoint(newFrameSet,copyed_pose);//현재 추정된 전역 카메라 포즈(estimated_pose)와 새로운 프레임에 대한 정보를 
                                                                  //기반으로 맵포인트를 추가합니다. 
            //Todo 시작 
            //# < 3 step > < 2 step > 전에 유지하던 로컬 키프레임들(clone_past_localframe)의 맵포인트에 대해서 생성이후에 지난 시간(make_count)를 업데이트하고,  
            //이후에 보였는지(view_count)를 기반으로 하여 맵포인트를 삭제합니다. 논문에서는 처음삽입이후에 3번이상 보였는지를 기반으로 삭제하는듯 합니다. 추가적으로 필요한 변수가 있는지 알려주세요.
            for(int i=0; i<clone_past_localframe.size(); i++)
            {
                KeyFrame* past_local_keyframe = clone_past_localframe[i];//로컬 키프레임을 얻어옵니다.
                map<int, MapPoint*> past_map_point = past_local_keyframe->Get_MapPoint();//해당 키프레임의 맵포인트를 얻어옵니다. 
                int k =0;
                for(auto it_point = past_map_point.begin(); it_point !=past_map_point.end(); it_point++)
                {
                    int point_idx_global = it_point->first;//해당 맵포인트에 대한 전역 인덱스입니다.
                    MapPoint* point_in_frame = it_point->second;//해당 맵포인트입니다. 
                    
                    if(point_in_frame->make_count +3 <newFrameSet->CurrentFrame->Get_KeyIndex() && point_in_frame->view_count <3)
                    {//
                        k++;
                        //맵포인트에 대해서 가지고 있는 키프레임들을 하나씩 지웁니다. 
                        map<int,int> key_matches = point_in_frame->keypoint_match;//맵포인트가 가지고 있는 키프레임 - 키포인트 인덱스관계입니다.
                        for(auto key_match = key_matches.begin(); key_match !=key_matches.end(); key_match++)
                        {
                            int keyframe_idx = key_match->first;
                            int keypoint_idx = key_match->second;

                            nodehandler._pt_KeyFrames[keyframe_idx]->_pmappoint_OwnedMapPoint.erase(point_in_frame->int_Node);
                            
                            //키프레임에서 해당 키인덱스 와 맵포인트 사이의 인덱스 
                            nodehandler._pt_KeyFrames[keyframe_idx]->_map_keyIdx2MapPointIdx.erase(keypoint_idx);
                        }
                    }
                }
            }
            //# < 4 step > 컬링이 완료된 로컬 맵포인트들과 현재 키프레임에 대해서 관측된 맵포인트들과 키프레임들사이의 R,t관계를 구하고, 최적화합니다. 화이팅
            vector<map_point_kch> InitialMapPointSet;
            map<int,key_frame_kch> InitialkeyFrameSet;
            
 
            Get_OptimizeSet(nodehandler._pt_LocalWindowKeyFrames, InitialMapPointSet,InitialkeyFrameSet);//초기 매칭할 수 있는 형태로 변화시켜줍니다. 
            Optimize_localmap(InitialMapPointSet,InitialkeyFrameSet);
            
            Set_OptimizeSet(nodehandler._pt_LocalWindowKeyFrames, InitialMapPointSet,InitialkeyFrameSet);//최적화된 키프레임과 맵포인트를 반영합니다. 
            
 
            
            //실제 반영하는 코드도 만들어야합니다. 
            nodehandler.v_newKeyFrame.erase(nodehandler.v_newKeyFrame.begin());//현재 키프레임 후보에서 방금 처리한 키프레임을 없앱니다. 
            nodehandler.estimated_pose.push_back(estimated_pose);//(그리는용도) 현재 추정된 위치를 그립니다. 
            //--실제 업데이트하는 부분 끝
            
            
        }
        nodehandler.m_sharedlock.unlock();//뮤텍스락으로 노드핸들링권한을 반납하여, tracking node가 일하게합니다.
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
      return_origin=(Mat_<double>(4,4) <<p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,0,0,0,1);//초기자세를 Mat의 형태로 얻습니다.
  }
  return return_origin;
}
Mat R_t_scale_2_Mat(double* R_t_scale)
{//프레임에 저장된 R,t변수를 카메라 외부행렬로 바꿉니다. 
    Mat Extrinsic_matrix = //카메라 외부 행렬을 가져옵니다.
            (Mat_<double>(4,4)    
            <<R_t_scale[0],R_t_scale[1],R_t_scale[2],R_t_scale[3],
              R_t_scale[4],R_t_scale[5],R_t_scale[6],R_t_scale[7],
              R_t_scale[8],R_t_scale[9],R_t_scale[10],R_t_scale[11],
                        0.,          0.,           0.,R_t_scale[12]);
    return Extrinsic_matrix;
}
void Get_OptimizeSet(vector<KeyFrame*> past_localframe, vector<map_point_kch> &InitialMapPointSet, map<int,key_frame_kch> &InitialkeyFrameSet)
{


    vector<int> localKeyFrameIdx; 
    for(int frame_ind = 0; frame_ind <past_localframe.size(); frame_ind++)
    {
        localKeyFrameIdx.push_back(past_localframe[frame_ind]->Get_KeyIndex());
    }
    vector<int> local_mappoint_idx;
    map<int, map_point_kch*> local_mappoint_saved;
    for(int reverse_frame_ind = past_localframe.size()-1; reverse_frame_ind >=0;reverse_frame_ind--)
    {//맨마지막에 있는 프레임부터 관계되어있는것을 하나둘 씩 미리 맵포인트를 만들어둡니다. 
        KeyFrame* revFrame = past_localframe[reverse_frame_ind];
        key_frame_kch newFrame;
        newFrame.frameNumber = revFrame->Get_KeyIndex();//초기 프레임번호를 초기화합니다.
        Mat globalpose = revFrame->global_pose;
        double x,y,z;
        x = globalpose.at<double>(0,3);
        y = globalpose.at<double>(1,3);
        z = globalpose.at<double>(2,3);
        newFrame.world_x_y_z = Point3d(x,y,z); //위치를 세팅합니다.
        Mat R;//글로벌방위를 담을 배열입니다.
        globalpose(Rect(0,0,3,3)).copyTo(R);//전역위치에서 글로벌 방위로 변환합니다.
        Mat rod_R(3,1,CV_64FC1);
        Rodrigues(R,rod_R);//로드리게스형으로 저장합니다.        
        newFrame.view_point = Point3d(rod_R.at<double>(0,0),rod_R.at<double>(0,1),rod_R.at<double>(0,2)); //방위를 초기화합니다.
        

        map<int,MapPoint*> have_mapset = revFrame->Get_MapPoint();//전역 맵포인트 인덱스(key)로 맵포인트의 포인터(value)를 가져옵니다.
        map<int,int> map_localIdx = revFrame->_map_keyIdx2MapPointIdx;//키프레임 내에서의 인덱스(key)를 전역 맵포인트 인덱스(value)으로 바꿉니다.
        
        for(auto idx = map_localIdx.begin(); idx != map_localIdx.end(); idx++)
        {
            int local_idx = idx->first;//지역 인덱스
            int global_idx = idx->second;//global Idx
            MapPoint* global_mapPoint = have_mapset[global_idx];
            bool isHave = find(local_mappoint_idx.begin(),local_mappoint_idx.end(),global_idx)==local_mappoint_idx.end();
            
            if(isHave)
            {//해당 맵포인트가 아직 없다면
                map_point_kch* new_mappoint = new map_point_kch;
                new_mappoint->mappointNumber = global_idx;//전역 인덱스
                new_mappoint->world_x_y_z =  global_mapPoint->p3d_coordinate;//3차원 좌표
                local_mappoint_saved[global_idx] = new_mappoint;
                local_mappoint_idx.push_back(global_idx);
                //현재키프레임에 해당하는 정보를 추가합니다.
                new_mappoint->frame2idx[newFrame.frameNumber] = local_idx;//특정 프레임(key)에서 이 맵포인트의 인덱스(value) 입니다. 
                newFrame.idx2Point2d[local_idx] = global_mapPoint->pixel_match[newFrame.frameNumber];
            }
            else
            {//해당맵포인트가 있다면
                map_point_kch* copyed_mappoint = local_mappoint_saved[global_idx];
                //현재키프레임에 해당하는 정보를 추가합니다.
                copyed_mappoint->frame2idx[newFrame.frameNumber] = local_idx;//특정 프레임(key)에서 이 맵포인트의 인덱스(value) 입니다. 
                newFrame.idx2Point2d[local_idx] = global_mapPoint->pixel_match[newFrame.frameNumber];
            }
        }
        InitialkeyFrameSet[revFrame->Get_KeyIndex()] = newFrame;//해당 로컬키프레임을 최적화를 위한 프레임으로 저장합니다.
    }
    std::transform(local_mappoint_saved.begin(),local_mappoint_saved.end(),std::back_inserter(InitialMapPointSet),[](auto &kv){return *(kv.second);});
    // cout<<InitialMapPointSet.size()<<endl;
}
void Set_OptimizeSet(vector<KeyFrame*> &past_localframe, vector<map_point_kch> InitialMapPointSet, map<int,key_frame_kch> InitialkeyFrameSet)
{
    map<int, map_point_kch> local_mappoint_saved;
    for(int i=0; i <InitialMapPointSet.size(); i++)
    {
        map_point_kch check_mapPoint = InitialMapPointSet[i];
        local_mappoint_saved[check_mapPoint.mappointNumber] = check_mapPoint;
    }
    vector<int> already_checked_mapPoint;

    for(int i=0; i<past_localframe.size(); i++)
    {
        KeyFrame* pFrame = past_localframe[i];
        key_frame_kch nFrame = InitialkeyFrameSet[pFrame->Get_KeyIndex()];//새롭게 수정된프레임
        Point3d new_ViewPoint =nFrame.view_point;
        Point3d new_World_x_y_z =nFrame.world_x_y_z;
        Mat R(3,3,CV_64FC1);
        Mat rod_R(3,1,CV_64FC1);
        Mat NewGlobalPose= Mat::zeros(4,4,CV_64FC1);
        rod_R.at<double>(0,0) = new_ViewPoint.x;
        rod_R.at<double>(1,0) = new_ViewPoint.y;
        rod_R.at<double>(2,0) = new_ViewPoint.z;
        Rodrigues(rod_R,R);
        R.copyTo(NewGlobalPose(Rect(0,0,3,3)));
        NewGlobalPose.at<double>(0,3) = new_World_x_y_z.x;
        NewGlobalPose.at<double>(1,3) = new_World_x_y_z.y;
        NewGlobalPose.at<double>(2,3) = new_World_x_y_z.z;
        NewGlobalPose.at<double>(3,3) = 1.0;
        pFrame->global_pose = NewGlobalPose;
        map<int,MapPoint*> origin_mappoint = pFrame->_pmappoint_OwnedMapPoint;
        for(auto mp = origin_mappoint.begin(); mp !=origin_mappoint.end(); mp++)
        {
            int mp_idx = mp->first;
            MapPoint* changed_mp = mp->second;
            
            bool isNewPoint = find(already_checked_mapPoint.begin(),already_checked_mapPoint.end(),mp_idx) ==already_checked_mapPoint.end();//이부분은 수정할것
            if(isNewPoint)
            {
                already_checked_mapPoint.push_back(mp_idx);
                map_point_kch NewMappoint = local_mappoint_saved[mp_idx];
                changed_mp->p3d_coordinate = NewMappoint.world_x_y_z;
            }
            else
            {
                continue;
            }
        }
    }
}