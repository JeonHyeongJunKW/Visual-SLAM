#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Node/NodeHandler.h"
using namespace std;
using namespace cv;
void LocalMapping(NodeHandler &nodehandler);
Mat load_origin_pose(const char* filename);
Mat R_t_scale_2_Mat(double* R_t_scale);
void Optimize_localmap();