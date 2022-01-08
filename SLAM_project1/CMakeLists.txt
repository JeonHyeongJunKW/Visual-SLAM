cmake_minimum_required(VERSION 3.5)

project(KeyFrameViewer)

find_package(OpenCV REQUIRED)
find_package(DBoW2 REQUIRED)
# helloworld
include_directories(${OpenCV_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_executable(KeyFrameViewer KeyFrameMakeAndVisualOdometry.cpp
                                Node/NodeHandler.cpp 
                                Node/NodeHandler.h 
                                Node/KeyFrame.h 
                                Node/KeyFrame.cpp
                                Node/CameraTool.h
                                Node/CameraTool.cpp
                                Tracking/Tracking.h
                                Tracking/Tracking.cpp
                                LocalMapping/LocalMapping.h
                                LocalMapping/LocalMapping.cpp)
target_link_libraries(KeyFrameViewer ${OpenCV_LIBS} ${DBoW2_LIBS})