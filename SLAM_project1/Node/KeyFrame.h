#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class MapPoint
{
  /*
  - 3차원의 전역계에 대한 좌표
  - 키프레임 - 2D좌표 정보 
  - 보여진 횟수 : 0
  - 생성 카운트 : 0
  - 최소, 최대 거리(d_max, d_min)
  - 스케일정보? ~~~~이부분은 Orb feature에 대해서 조사해봐야할듯
  - Descriptor 정보 
  - 노드 인덱스 
  */
  public:
    Point3d p3d_coordinate;//3차원점을 가지고 있습니다. 
    Mat mat_Orbdescriptor;// 해당 맵포인트가 가진 Orb descriptor 정보입니다.1x32 크기입니다.
    map<int,Point2d> pixel_match;//특정 프레임과 픽셀좌표 사이의 관계
    int view_count =0;//해당 맵포인트가 보여진횟수
    int make_count = 0;//해당 점이 생긴지 지난횟수입니다. 이 횟수가 3일때, 
                      //view count가 3보다 작다면 이 맵포인트는 지워집니다 . 
    int max_d; //해당 점이 보여질 수 있는 최대거리
    int min_d; //해당 점이 보여질 수 있는 최소거리
    map<int,int> octave_match;//각 키프레임에서 보이는 옥타브정보 
    int int_Node = 0;//노드의 이름 
};

class KeyFrame
{
  public://variable
    int _int_Keyindex;//키프레임의 고유 번호입니다.
    Mat _mat_Intrinsicparam;//내개 파라미터 
    Mat _mat_Originimage;//원본 이미지입니다. (내개 파라미터 미적용)
    map<int, MapPoint*> _pmappoint_OwnedMapPoint;//해당 키프레임이 가지고 있는 맵포인트들의 집합입니다. 포인터로 관리합니다. 
    map<int, int> _map_keyIdx2MapPointIdx;//키포인트의 인덱스에 대해서 맵포인트상의 인덱스로 변환합니다. 키값이 존재하지않으면 새로배정합니다.
    KeyFrame* _pkeyframe_Fatherkeyframe;//이 키프레임의 이전 키프레임의 포인터입니다. 키프레임간에 유사도 검사등에 사용됩니다.
    KeyFrame* _pkeyframe_Childkeyframe;//이 키프레임의 다음 키프레임의 포인터입니다. 키프레임간에 유사도 검사등에 사용됩니다.
    double* _pf_camera_R_t;//12개의 파라미터이다.[R | T]가 행단위로 들어가 있다. 추가로 마지막 성분으로 스케일이 들어간다.
    
    
  public: //constructor method
    KeyFrame(int arg_KeyIndex);
    KeyFrame(int arg_KeyIndex, Mat arg_IntrinsicParam);
    KeyFrame(int arg_KeyIndex, Mat arg_IntrinsicParam, Mat arg_OriginImage);
    KeyFrame(int arg_KeyIndex, Mat arg_IntrinsicParam, Mat arg_OriginImage,double* arg_camera_R_t);

  public: //method
    //Intrinsic Parameter
    //각 프레임의 디스크립터와 키포인트를 등록합니다.
    Mat _mat_descriptors;
    Mat global_pose;
    vector<KeyPoint> _vkey_keypoints;
    void Set_IntrinsicParam(Mat arg_IntrinsicParam); // 키프레임의 내계 파라미터를 바꿉니다. 
    Mat Get_IntrinsicParam(void); // 키프레임의 현재 내계 파라미터를 반환합니다.

    //Origin Image
    void Set_OriginImage(Mat arg_OriginImage); // 키프레임의 원본이미지를 바꿉니다.
    Mat Get_OriginImage(void); // 키프레임의 원본이미지를 가져옵니다.

    //Key Index
    int Get_KeyIndex(void); //키프레임의 고유 인덱스를 가져옵니다. 

    //neighbor keyFrame
    KeyFrame* Get_FatherKeyFrame(void);//부모 키프레임을 반환합니다.
    void Set_FatherKeyFrame(KeyFrame* arg_FatherFrame);//부모 키프레임을 반환합니다.
    KeyFrame* Get_ChildKeyFrame(void);//자식 키프레임을 반환합니다.
    void Set_ChildKeyFrame(KeyFrame* arg_ChildFrame);//부모 키프레임을 반환합니다.

    map<int, MapPoint*> Get_MapPoint(); // 현재 키프레임이 가지고 있는 맵포인트를 반환합니다. 
    int Get_NumMapPoint(); //현재 가지고 있는 맵포인트 

    void Add_MapPoint(MapPoint* arg_MapPoint);//키프레임에 맵포인트를 추가합니다. 
    void Set_Descriptor(Mat arg_descriptors);
    void Set_KeyPoint(vector<KeyPoint> arg_keyPoint);
    Mat Get_Descriptor(void);
    vector<KeyPoint> Get_keyPoint(void);
    void Set_Rt(Mat arg_R, Mat arg_T);
    void Get_Rt(double * &R_tparam)
    {
      R_tparam =this->_pf_camera_R_t;
    }
};