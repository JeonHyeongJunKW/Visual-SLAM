cmake_minimum_required (VERSION 3.5) 		
project(myproject)					
 
 
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cc)
 
add_executable(main main.cc)   

find_package(Ceres REQUIRED)		    		  
find_package(OpenCV REQUIRED)		    		
find_package(DBoW2 REQUIRED)

target_link_libraries(main ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

file(GLOB SOURCES *.cpp) 
include_directories(${OpenCV_INCLUDE_DIR} ${DBoW2_INCLUDE_DIRS})

target_link_libraries(main Ceres::ceres)	
target_link_libraries(main ${OpenCV_LIBRARIES} ${DBoW2_LIBS})

