cmake_minimum_required(VERSION 3.5)

project(KeyFrameViewer)

find_package(OpenCV REQUIRED)
find_package(DBoW2 REQUIRED)
# helloworld
include_directories(${OpenCV_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS})
add_executable(KeyFrameViewer KeyFrameMakeAndRelocalization.cpp Node/NodeHandler.cpp Node/NodeHandler.h Node/KeyFrame.h Node/KeyFrame.cpp)
target_link_libraries(KeyFrameViewer ${OpenCV_LIBS} ${DBoW2_LIBS})