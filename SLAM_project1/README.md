# SLAM project 1

## 목표 
: 모노 카메라 기반의 SLAM구현 

## Step 1 :  데이터셋에서 ORB feature 추출 및 key frame 인지
데이터셋 : KITTI dataset중에서 0번의 왼쪽카메라 데이터셋 사용 
- VisualizeOdometry.cpp : 실제 정답 Odometry를 시각화합니다. 주의) 행의 좌표는 확대해서 그립니다.

<img width ="600" src="https://user-images.githubusercontent.com/63538314/146316571-60f79765-8e3e-4506-af88-35fc96511184.gif">

``` c++
//이미지 크기 결정
int image_width = (int)(max_point_x- min_point_x+20);//좌우로 10칸씩 추가하였다. 
int image_height = (int)(max_point_y - min_point_y)*10+20;//좌우로 10칸씩 추가하였다. 행은 10배로 늘렸다.

//이미지 픽셀 수정하기 
int col = 10+(points[i].x-min_point_x);//위쪽 10칸
int row = 10+((points[i].y-min_point_y))*10;//왼쪽 10칸씩 추가 행은 10배로 늘렸다.
data[row*map_image.cols + col] = Vec3b(0,0,255);
```



