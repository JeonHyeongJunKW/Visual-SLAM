cmake_minimum_required (VERSION 3.5) 			 # CMake 사용하는 최소 버전 정하기
project(myproject)					 # 내가 만들 프로젝트 이름(주석느낌)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)
find_package(OpenCV REQUIRED)		    		 # 쓰고싶은 패키지 찾는 것
 
file(GLOB SOURCES *.cpp) 
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(main main.cc)   		 # 실행 파일로 만들 파일
target_link_libraries(main ${OpenCV_LIBRARIES})
