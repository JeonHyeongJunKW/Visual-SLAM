#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
  
  vector<std::vector<cv::Point3f> > objpoints;//3차원 점의 벡터 (2차원집합)
  vector<std::vector<cv::Point2f> > imgpoints;//픽셀좌표의 2차원 벡터 
  vector<Point3f> objp; //3차원 점의 chessboard상의 좌표 (이미 정해져있다.)
  //체스보드에서 한칸씩 빼야한다.
  for(int i =0; i<6;i++)
  {
    for(int j=0; j<8;j++)
    {
      objp.push_back(Point3f(j,i,0));
    }
  }
  
  int img_width =0;
  int img_height = 0;
  cout<<"initialize"<<endl;
  for( int i =0; i <13; i++)
  {
    ostringstream name;
    vector<Point2f> corner_pts;
    name<< "./chessboard/board"<<i<<".png"<<endl;
    Mat chess_img = imread(name.str(),IMREAD_GRAYSCALE);
    
    bool success;
    success = findChessboardCorners(chess_img,Size(8,6),corner_pts,CV_CALIB_CB_ADAPTIVE_THRESH |CV_CALIB_CB_FAST_CHECK| CV_CALIB_CB_NORMALIZE_IMAGE);
    img_width = chess_img.cols;
    img_height = chess_img.rows;
    if(success)//성공했다면 
    {
      cout<<"find chessboard"<<endl;
      TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
      cornerSubPix(chess_img,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
      // drawChessboardCorners(frame, cv::Size(7,5), corner_pts, success);
      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }
    
  }
  
  Mat cameraMatrix,distCoeffs,R,T;
  calibrateCamera(objpoints, imgpoints, cv::Size(img_height,img_width), cameraMatrix, distCoeffs, R, T);
  //실제 내계파라미터를 확인합니다.
  cout<<cameraMatrix<<endl;
  cout<<distCoeffs<<endl;
  cout<<R<<endl;
  cout<<T<<endl;
  FileStorage fs_w("C_MatrixNdisCoeffs.xml",FileStorage::WRITE);
  fs_w <<"cameraMatrix"<<cameraMatrix;
  fs_w<<"disCoeffs"<<distCoeffs;
  fs_w.release();
  /*
  다시 읽을 때는 
  FileStorage fs_r("C_MatrixNdisCoeffs.xml",FileStorage::READ);
  fs_r["cameraMatrix"] >>cameraMatrix;
  fs_r["disCoeffs"] >>distCoeffs;
  fs_r.release();
  */


  return 0;
}