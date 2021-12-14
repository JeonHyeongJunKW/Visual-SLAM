#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
  VideoCapture cap(0);
  Mat img;
  int image_index = 0;
  int keycode = 'n';
  cout<<"start capture"<<endl;
  while ((keycode = waitKey(1)) != 37)
  {
    cap >>img;
    if(keycode =='c')
    {
      cap >>img;
      ostringstream name;
      name<< "board"<<image_index<<".jpg"<<endl;
      cout<<"saving"<<endl;
      imwrite(name.str(),img);
      image_index++;
    }
    imshow("board",img);
    
  }
  return 0;
}