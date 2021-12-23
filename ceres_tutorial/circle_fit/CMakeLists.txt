cmake_minimum_required (VERSION 3.5) 			 # CMake 사용하는 최소 버전 정하기
project(myproject)					 # 내가 만들 프로젝트 이름(주석느낌)
 
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES circle_fit.cpp)
find_package(Ceres REQUIRED)		    		 # 쓰고싶은 패키지 찾는 것
find_package(OpenCV REQUIRED)
 
file(GLOB SOURCES *.cpp) 
include_directories(${OpenCV_INCLUDE_DIR})
	 
add_executable(circle_fit circle_fit.cc)   
		
target_link_libraries(circle_fit ${OpenCV_LIBRARIES})
target_link_libraries(circle_fit Ceres::ceres gflags glog)  # 라이브러리 등록
