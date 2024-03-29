# SLAM project 1

## 목표 
: 모노 카메라 기반의 SLAM구현 

## Step 1 :  데이터셋에서 ORB feature 추출 및 Keyframe 인지[형준]
데이터셋 : KITTI dataset중에서 0번의 왼쪽카메라 데이터셋 사용 
특징 : 
  - 10fps단위로 찍힌 시퀀스이다. 


- VisualizeOdometry.cpp : 실제 정답 Odometry를 시각화합니다. 주의) 행의 좌표는 확대해서 그립니다. (x,y 좌표를 시각화함)(21-12-16)

<img width ="600" src="https://user-images.githubusercontent.com/63538314/146316571-60f79765-8e3e-4506-af88-35fc96511184.gif">

- LoadGrayImage.cpp : 이미지를 glob으로 읽어서 시각화합니다.(21-12-18)

- KeyFrameMaker.cpp : 이미지를 glob으로 읽어와서 orb feature를 뽑습니다. 
	- 키프레임으로 적합한지 검사합니다.(21-12-19)
	- 키프레임에 대해서 맵포인트를 구합니다.(21-12-20) (미완 : 초기 키프레임 삽입간에 2개 이상의 점이 나오지 않는경우와 유사도의 임계값 및 순간적인 우선순위가 안정해짐)
  - 키프레임에 대해서 맵포인트 구하기, 완성 하지만 맵포인트가 과하게 많아지므로 불필요하게 많은 맵포인트가 발생, 어느정도 가지치기하는 것이 필요.(맵포인트의 3차원 위치와 카메라 위치 고려)
  
- KeyFrameViewer.cpp : 키프레임과 맵포인트를 얻습니다. 
  - 로컬맵에서 유지하는 프레임은 고정된 갯수로하였다.

- ~~KeyFrameMakeAndRelocalization.cpp : 키프레임 및 맵포인트를 생성하고, 트랙킹이 실패할경우 리로컬라이제이션을 넣음.~~

- KeyFrameMakeAndVisualOdometry.cpp : 키프레임 및 맵포인트를 생성하고, homography와 fundamental matrix를 기반으로 odoemtry를 생성

  ### 목표 
  - NodeHandlerWithStatic.cpp
    - 기존에 정적인 프레임 수를 저장하는방식입니다.
    
  - NodeHandler.cpp 
    - 현재 프레임에서 얻은 맵포인트를 포함하는 키프레임(K1)을 로컬맵에서 유지하는 키프레임으로 설정한다. 
    - 현재 프레임과 가장 많은 맵포인트를 공유하는 키프레임(K_ref)를 유지한다. 

  - Tracking.cpp 
    - 카메라 초기위치 추정, 키프레임 후보 여부 검사를 하고 있음.
    - (추가예정) Global relocalization

  - localMapping.cpp
    - 키프레임에 대한 DBOW 등록 및 local bundle adjustment수행

  - ~~DBoW2 : 기존의 ORB feaure의 Database 및 Visual word를 사용할 수 있게 하였다. (21-12-27)~~ 


    
  <img width ="600" src="https://user-images.githubusercontent.com/63538314/147348554-440c3006-0f46-4c82-a7d3-51ed69bc46b9.gif">
  
  : 빨간색은 키포인트들의 좌표, 파란색은 현재 키포인트에 대하여 맵포인트를 공유하는 키프레임(k1) (x,y 좌표를 시각화함) (21-12-24)
  -> 아래와 같이 변경 

  ![real_last](https://user-images.githubusercontent.com/63538314/148644845-8c6f8d6f-99ab-406a-876d-9db0faa7f7bf.gif)

  - 빨간색은 ground truth 의 좌표, 파란색은 Visual odometry로만 추정된 좌표 (x,z 좌표를 시각화함) (22-01-08)

  - Tracking/Tracking.cpp
    - tracking을 하기위한 Visual odometry의 R,t초기화 구현됨

  - LocalMapping/LocalMapping.cpp
    - 키프레임 등록 및 odometry업데이트까지됨
    - 맵포인트 등록 및 불완전한 맵포인트 선별 및 제거코드 구현
  
### KeyFrame이 가진 정보 [형준]
- 카메라 포즈 및 내부 파라미터
- 해당 프레임이 가진 orb feature list

### KeyFrame 추출방식 목표 [형준]
- 전역 초기화후에 20프레임이 넘었는지 
- 로컬 매핑이 쉬고 있거나 마지막 key frame이 삽입된지 20 frame이 넘었는지
- 현재의 키프레임이 적어도 50개의 orb feature를 가지고 있는지
- 후보 프레임이 ~~다른 키프레임~~ 키프레임(K_ref)보다 90프로 이하로 다른 점들을 트랙킹하는지(유사도가 다소 낮은지 확인하는것)

## Step 2 : KeyFrame에서 map point 생성 및 제거 [형준]

## Step 3 : covisibility graph 구현 [찬혁]
구현내용 : 
  - 각 KeyFrame의 위치 및 방위 정보 노드화
  - map point의 위치도 최적화 대상임, 이것도 노드형태로 유지 
  - 
## Step 4 : Essential graph 구현 [찬혁]
구현내용 : 
  - Covisibility graph에 추가로 spanning tree를 만들고, Covisibility간에 약한건 지움
## step 5 : local BA 구현 [찬혁]
구현내용 : 
  - Covisibility graph내에서 새로운 key Frame 추가간에 노드 정보 유지 

## step 6 : loop closing 인지/ DBOW 유지및 관리 [형준], loop closing 수행(Essential graph 최적화)[찬혁]

# 1차 목표기간 :  6주

## Update 내역 
21/01/30 - 맵포인트 추출 및 제거 알고리즘 구현, 추후에 최적화 진행예정, 키프레임 결정 알고리즘 보완예정