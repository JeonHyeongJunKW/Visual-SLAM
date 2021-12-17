#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

class MapPoint
{
  public:
    Point3f p3f_coordinate;//3차원점을 가지고 있습니다. 
    vector<KeyFrame*> vec_pt_ObservedKeyFrame; //해당 map point가 포함된 keyframe들
    map<KeyFrame*,int> map_pointInd; //map point가 소속된 키프레임에서 해당 map point의 인덱스
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
    float* _pf_camera_R_t;

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
    
    //MapPoint
    void Make_MapPoint();//기존에 가지고 있던 원본이미지에서 맵포인트를 만듭니다.
    void Make_MapPoint(Mat arg_OriginImage);//지우고 인자로 받은 이미지에서 맵포인트를 만듭니다. 

    vector<MapPoint*> Get_MapPoint(); // 현재 키프레임이 가지고 있는 맵포인트를 반환합니다. 
    int Get_NumMapPoint(); //현재 가지고 있는 맵포인트 
};