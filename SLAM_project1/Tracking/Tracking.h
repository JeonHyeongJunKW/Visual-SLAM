#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Node/NodeHandler.h"


using namespace std;
using namespace cv;
void tracking_thread(NodeHandler& nodehandler);
bool GetInitialRt(NodeHandler &nodehandler,
                    KeyFrame* kfp_NewFrame,
                    KeyFrame* kfp_LastFrame, 
                    Mat &R, 
                    Mat &t);
bool track_localMap(NodeHandler &nodehandler, KeyFrame* kfp_NewFrame,vector<Match_Set> &localmap);//R,t 추정후에 기존에 가진 맵포인트를 비교한다.근데 없다. 넣어줘야 비교하지..
bool decide_newkeyframe(NodeHandler &nodehandler, KeyFrame* kfp_NewFrame);//새로운 키프레임으로 쓸지를 정한다.