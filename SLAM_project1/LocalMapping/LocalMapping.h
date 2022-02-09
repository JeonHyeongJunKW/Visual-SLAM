#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Node/NodeHandler.h"
using namespace std;
using namespace cv;
struct map_point_kch
{
    int mappointNumber;//맵포인트의 인덱스
    Point3d world_x_y_z;//월드계기준의 맵포인트 좌표
    map<int, int> frame2idx;//특정 프레임(key)에서 이 맵포인트의 인덱스(value) 입니다. 
    
};

struct key_frame_kch
{
    int frameNumber;//프레임의 이름
    Point3d world_x_y_z;//월드계기준의 키프레임의 중심좌표
    Point3d view_point;//월드계기준의 키프레임의 방위 
    map<int,Point2d> idx2Point2d;//특정 키포인트의 좌표(x,y : value)입니다. 키프레임내에서의 인덱스(key)로 유지됩니다. 
    map<int, Mat> idx2Descriptor;//특정 키포인트의 descriptor입니다. 키프레임내에서의 인덱스(key)로 유지됩니다. 
};

void LocalMapping(NodeHandler &nodehandler);
Mat load_origin_pose(const char* filename);
Mat R_t_scale_2_Mat(double* R_t_scale);
void Optimize_localmap( vector<map_point_kch> &MapPointSet, map<int,key_frame_kch> &keyFrameSet);//로컬 키프레임 셋 내에서의 상호관계입니다. 각 일련의 맵포인트들을 가지고, 매칭셋을 구해서 만들어야합니다. 
void Get_OptimizeSet(vector<KeyFrame*> past_localframe, vector<map_point_kch> &InitialMapPointSet, map<int,key_frame_kch> &InitialkeyFrameSet);
void Set_OptimizeSet(vector<KeyFrame*> &past_localframe, vector<map_point_kch> InitialMapPointSet, map<int,key_frame_kch> InitialkeyFrameSet);