cmake_minimum_required (VERSION 3.5) 		
project(myproject)					
 
 
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES covisivility.cc)
 
add_executable(covisivility covisivility.cc)   

find_package(Ceres REQUIRED)		    		  
find_package(OpenCV REQUIRED)		    		
 
file(GLOB SOURCES *.cpp) 
include_directories(${OpenCV_INCLUDE_DIR})

target_link_libraries(covisivility Ceres::ceres)	
target_link_libraries(covisivility ${OpenCV_LIBRARIES})	
