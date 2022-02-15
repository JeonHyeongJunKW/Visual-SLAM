/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <mutex>
#include <condition_variable>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


using namespace DBoW2;
using namespace cv;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 100; //  i < 4540 + 1 = SIMAGES + NIMAGES
const int SIMAGES = 1;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

vector<vector<cv::Mat> > features;
vector<vector<cv::Mat> >* pfeatures = &features;
// ----------------------------------------------------------------------------

void show_(Mat& i1, Mat& i2);

class ODE
{
public:
  vector<Point2f> points;
  Mat map_image;
  Mat map_image2;
  Vec3b* data = nullptr;  // 배경(빨간 실선)
  Vec3b* data2 = nullptr; // 전경(초록 실선)
  const Vec3b red = Vec3b(0,0,240), blue = Vec3b(0, 240, 0);
  const int gap = 10, scale_x = 1, scale_y = 10;
  int number_of_points = -1;
  int image_width = -1, image_height = -1;
  std::thread th;
  std::mutex g_mutex;
  std::condition_variable g_controller;

  ODE(const string filename_odome){
    ExtractPoint2D(filename_odome, this->points);
    initOdometry2D(this->points);
  };

  void initOdometry2D(vector<Point2f>& points)
  {
    float max_point_x = -10000;
    float min_point_x = 10000;
    float max_point_y = -10000;
    float min_point_y = 10000;

    for(const auto& point : points){
      max_point_x = max(point.x, max_point_x);
      min_point_x = min(point.x, min_point_x);

      max_point_y = max(point.y, max_point_y);
      min_point_y = min(point.y, min_point_y);
    }
    this->image_width = (max_point_x - min_point_x) * scale_x + gap * 2;         //좌우로 10 pixel씩 추가하였다. 
    this->image_height = (max_point_y - min_point_y) * scale_y + gap * 2;    //좌우로 10칸씩 추가하였다. 행은 10배로 늘렸다.

    cout << " set image size : " << image_height << ", " << image_width << "\n";

    this->map_image = Mat::zeros(image_height, image_width, CV_8UC3);
    this->data = (Vec3b*)map_image.data;
    this->map_image2 = Mat::zeros(image_height, image_width, CV_8UC3);
    this->data2 = (Vec3b*)map_image2.data;
    
    cout << "minimum x : " << min_point_x << ", minimum y : " << min_point_y << "\n"
        << "maximum x : " << max_point_x << ", maximum y : " << max_point_y << "\n"
        << "width : " << image_width << " height : " << image_height << "\n";

    for(auto& point : points){
      point.x = gap + (point.x - min_point_x) * scale_x;
      point.y = gap + (point.y - min_point_y) * scale_y;
    }
    cout << " end initial ode \n";
  }

  // Index to Point
  int I2P(int index) const{
    // tdbcout << this->points[index].y << " / " << this->points[index].x << endl;
    return (int)this->points[index].y * this->image_width + (int)this->points[index].x;
  }

  // automatic set red odometry point
  bool setPoint(){
    static int index = 0;     // index of points
    if(index >= number_of_points){
      cout << "set last point \n";
      return 0;
    }
    data[I2P(index++)] = red;
    // cout << I2P(index++) << endl;
    return 1;
  }

  // set blue odometry point
  bool setPoint(int index, bool color){
    if(index >= number_of_points){
      cout << " function error setPoint :: over than number of points \n";
      return 0;
    }
    
    if(color){
      data2[I2P(index)] = blue; // 사실 초록임, 보이는 건 노랑
    }
    else{
      data[I2P(index)] = red;
    }
    return 1;
  }
/*
  void show_thread(){
    // while(1){
    //   //std::unique_lock<std::mutex> lock(g_mutex);
    //   //g_controller.wait(lock); 
    //   this->show();
    // }

    this->th = thread(show_, this->map_image, this->map_image2);
  }*/

  void show(){
    imshow("ground true", this->map_image+this->map_image2);
    waitKey(1);
  }

  void show_last(){
    imshow("ground true", this->map_image+this->map_image2);
    waitKey(1);
  }

  // 입력된 파일명의 폴더에서 point를 파싱
  void ExtractPoint2D(const string& filename, vector<Point2f>& extracted_point2d){
    extracted_point2d.clear();//기존에 들어있는 점을 지웁니다.
    ifstream poseFile(filename);
    cout << " target file name : "<< filename << endl;

    string line_;
    float p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12;
        
    if(!poseFile.is_open()){
      cout << "function error : ExtractPoint2D : File is not opened !!" << endl;
    }

    while(!poseFile.eof()){
      getline(poseFile, line_);
      sscanf(line_.c_str(),"%e %e %e %e %e %e %e %e %e %e %e %e",&p1,&p2,&p3,&p4,&p5,&p6,&p7,&p8,&p9,&p10,&p11,&p12);
      extracted_point2d.push_back(Point2f(p4,p8)); //x,y 좌표
    }

    this->number_of_points = extracted_point2d.size();
    cout << " end file parsing \n";
  }
};

// ----------------------------------------------------------------------------

string makeImgName(const int index){
  stringstream ss;
  string snum = to_string(index);
  int num_of_zeros = 6-snum.size();
  ss << "/home/chan/kitti_dataset/data_odometry_gray/dataset/sequences/00/image_0/" ;
  for(int ii = 0; ii < num_of_zeros; ii++){
    ss << "0";
  }
  ss << snum << ".png";
  return ss.str();
}

bool loadFeatures(vector<cv::Mat >& feature, const int index)
{
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  // cout << "Extracting ORB features..." << endl;
  
  cv::Mat image = cv::imread(makeImgName(index), 0);
  cv::Mat mask;
  vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  if(image.empty()){
    // 이제는 에러가 아님
    cout << " functiom loadFeatures end :: load all image !! \n";
    return 0;
  }
  orb->detectAndCompute(image, mask, keypoints, descriptors);
  //imshow("t", image);
  //waitKey(1);

  //features.push_back(vector<cv::Mat >());
  changeStructure(descriptors, feature);
  image.~Mat();
  return 1;
  // cout << makeImgName(index) << '\n';
  // for(auto& i : features.back()){
  //   cout << i << endl;
  // }
}


void setDB(vector<BowVector>& db, OrbVocabulary& voc, vector<vector<cv::Mat> > &features){
  voc.create(features);
  for(auto& f : features){   
    db.push_back(BowVector());   
    voc.transform(f, db.back());
  }
}

// 자신의 앞 index까지만
const double DBoW_Theshold = 0.48;
const int min_loop_gap = 500;
int Database(vector<BowVector>& db, OrbVocabulary& voc, vector<vector<cv::Mat> > &features, const int target_index){
  double max_score = 0.;
  int max_index = 0, index = 0, max_id = -1;

  // add images to the database.
  auto iter = db.begin();
  auto end = iter + max(0, target_index - min_loop_gap);
  auto* target = &db.at(target_index);

  for(; iter != end; iter++, index++){
      double score = voc.score(*iter, *target);
      // cout << score << " ";
      if(max_score < score){
        max_score = score;
        max_id = index;
      }
  }
  
  cout << target_index << " : " << max_score << " " << max_id << endl;
  if(max_score > DBoW_Theshold){
    return 1;
  }

  return 0;
}

// ----------------------------------------------------------------------------
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------
OrbVocabulary intiDBoW(){
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  return OrbVocabulary(k, L, weight, scoring);

  //cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  //voc.create(features);
  //cout << "... done!" << endl;

  //cout << "Vocabulary information: " << endl
  //<< voc << endl << endl;

}


void VocCreation(const vector<vector<cv::Mat >> &features, OrbVocabulary& voc)
{
  voc.create(features);
  // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  cout << "v1 : ";
  
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
  
    for(int j = i+100; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      if(score > 0.4){
        // cout << "Image " << i << " vs Image " << j << ": " << score << endl;
      }
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

int testDatabase(OrbDatabase& db, vector<cv::Mat > &feature,  OrbVocabulary& voc)
{
  db.add(feature);

  QueryResults ret;
  db.query(feature, ret, 4);

  // ret[0] is always the same image in this case, because we added it to the 
  // database. ret[1] is the second best match.

  cout << "Searching for Image " << ret << endl;
  return 1;
}

// ----------------------------------------------------------------------------

void loadFeatures2(int s, int e, const int num_image);
void th_Test(int i, int ii);

int nth = std::thread::hardware_concurrency() * 2 + 1;

int main()
{

  vector<std::thread> vth;

  /*
  example) 
  /home/chan/kitti_dataset/data_odometry_gray/dataset/sequences/00/image_0/000100.png
  /home/chan/kitti_dataset/data_odometry_poses/dataset/poses/00.txt
  */
  string filename_odome = "/home/chan/kitti_dataset/data_odometry_poses/dataset/poses/00.txt";
  string filename_frame = "/home/chan/kitti_dataset/data_odometry_gray/dataset/sequences/00/image_0/000100.png";
  const int num_of_image = 4540;
  double nth_image = (double)num_of_image / (double)nth;

  ODE ode(filename_odome);

  //pfeatures->resize(num_of_image, vector<Mat>());
  for(int i = 0; i < num_of_image+1; i++){
    pfeatures->push_back(vector<Mat>());
  }
  OrbVocabulary voc = intiDBoW();
  // OrbDatabase db(voc, false, 0);
  vector<BowVector> db;
  int key1 = 1, key2 = 1, index = 0;

  for(double i = 0; i < num_of_image; i = i + nth_image){
    thread t1(loadFeatures2, i, i+nth_image, num_of_image);
    vth.push_back(move(t1));
    index++;
  }
  
  cout << nth << " / " << vth.size() << "개 의 thread를 사용합니다. \n";
  for(auto& th : vth){
    th.join();
  }

  cout << "...done :: end load feature ! \n";
  setDB(db, voc, features);
  cout << "...done :: end make DB ! \n";

  index = db.size() - 1;
  while(key2 && index){
    int flag = Database(db, voc, features, index);
    key2 = ode.setPoint(index, flag);  
    ode.show();
    index--;
  };

  cout << "프로그램 종료 \n";

  while(1){
    ode.show_last();
  }

  return 0;
/*
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  testVocCreation(features);

  wait();

  testDatabase(features);
*/
}

void loadFeatures2(int s, int e, const int num_image){
  e = min(e, num_image);
  while(s < e){
    loadFeatures(pfeatures->at(s), s);
    s++;
  }
}

void show_(Mat& i1, Mat& i2){
  while (true) {
    imshow("ground true", i1+i2);
    waitKey(1);
  }
}
/*
void printMessage(const std::string& messgae, int a);


int main() {
  
  vector<thread> v;
  std::thread t(printMessage, "thread test source", 1);

  v.push_back(move(t));
  v[1].join();

  return 0;
}

void printMessage(const std::string& message, int a)
{ 
  for(auto s : message)
    std::cout << s;
  std::cout << message << std::endl;
}
*/