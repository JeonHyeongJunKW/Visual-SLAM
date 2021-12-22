#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class MapPoint
{
  public:
    float* p3f_coordinate;//3차원점을 가지고 있습니다. 
    Mat mat_Orbdescriptor;// 해당 맵포인트가 가진 Orb descriptor 정보입니다.1x32 크기입니다.
    int int_Node;


    vector<int> vec_pt_ObservedKeyFrame; //해당 map point가 포함된 키프레임들
    map<int,KeyPoint> map_KeyPointInKeyFrame;//map point가 포함된 키프레임에서의 키포인트 정보 
    map<int,int> map_pointInd; //map point가 소속된 키프레임에서 해당 map point의 인덱스
};

class KeyFrame
{
  private://variable
    int _int_Keyindex;//키프레임의 고유 번호입니다.
    Mat _mat_Intrinsicparam;//내개 파라미터 
    Mat _mat_Originimage;//원본 이미지입니다. (내개 파라미터 미적용)
    vector<MapPoint*> _pmappoint_OwnedMapPoint;//해당 키프레임이 가지고 있는 맵포인트들의 집합입니다. 포인터로 관리합니다. 

    KeyFrame* _pkeyframe_Fatherkeyframe;//이 키프레임의 이전 키프레임의 포인터입니다. 키프레임간에 유사도 검사등에 사용됩니다.
    KeyFrame* _pkeyframe_Childkeyframe;//이 키프레임의 다음 키프레임의 포인터입니다. 키프레임간에 유사도 검사등에 사용됩니다.
    float* _pf_camera_R_t;//12개의 파라미터이다.[R | T]가 행단위로 들어가 있다. 추가로 마지막 성분으로 스케일이 들어간다.

  public: //constructor method
    KeyFrame(int arg_KeyIndex);
    KeyFrame(int arg_KeyIndex, Mat arg_IntrinsicParam);
    KeyFrame(int arg_KeyIndex, Mat arg_IntrinsicParam, Mat arg_OriginImage);
    KeyFrame(int arg_KeyIndex, Mat arg_IntrinsicParam, Mat arg_OriginImage,float* arg_camera_R_t);

  public: //method
    //Intrinsic Parameter
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

    vector<MapPoint*> Get_MapPoint(); // 현재 키프레임이 가지고 있는 맵포인트를 반환합니다. 
    int Get_NumMapPoint(); //현재 가지고 있는 맵포인트 

    void Add_MapPoint(MapPoint* arg_MapPoint);//키프레임에 맵포인트를 추가합니다. 
    
};