#include <iostream>
#include "Node/NodeHandler.h"
#include <opencv2/opencv.hpp>
#include "DBoW2.h"
#include "Node/CameraTool.h"
#include "Tracking/Tracking.h"
#include "LocalMapping/LocalMapping.h"
#include <thread>
#include <mutex>
using namespace cv;
using namespace std;
using namespace DBoW2;

const int k = 9;
const int L = 3;
const WeightingType weight = TF_IDF;
const ScoringType scoring = L1_NORM;


OrbVocabulary voc(k, L, weight, scoring);


int main(int argc, char** argv)
{
  char filename[] = "/media/jeon/T7/Kitti dataset/data_odometry_poses/dataset/poses/00.txt";
  vector<Point2f> point2d;
  cout << "Vocabulary information: " << endl<< voc << endl;
  int image_width;
  int image_height;
  float min_point_x;
  float min_point_y;
  ExtractPoint2D(filename, point2d, image_width, image_height, min_point_x,min_point_y);//file 정보 및 이미지의 높이를 가져온다.
  NodeHandler nodehandler(2,30);
  nodehandler.Set_InstricParam(718.856,718.856,0.00,607.1928,185.2157);

  thread t_thread(tracking_thread,ref(nodehandler));
  thread l_thread(LocalMapping,ref(nodehandler));
  t_thread.join();
  l_thread.join();
  
}