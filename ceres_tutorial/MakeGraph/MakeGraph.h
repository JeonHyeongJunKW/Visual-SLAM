#ifndef MakeGraph_chan
#define MakeGraph_chan

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <string>

// OpenCV을 사용하여 Graph를 그리는 헤더파일
// Data Visualization
// date : 21.12.22

namespace chan
{   
    namespace Graph_{
        // NDS 좌표계를 사용, z = 0
        // 10개의 data를 허용
        class Graph
        {
        private:
            std::vector<std::vector<cv::Point2d>*> vpvec;
            cv::Mat* frame = nullptr;
            int num_of_window = 0;
            void init_MakeGraph(){
                
            }

        public:
            const double map_size_x = 1000.;
            const double map_size_y = 1000.;
            const int circle_size = 3;
            const cv::Point2d center = cv::Size2d(map_size_x/2., map_size_y/2.);
            const cv::Size2i map_size = cv::Size2i(map_size_x, map_size_y);
            const cv::Scalar map_backgraund_color = cv::Scalar::all(230);

            cv::Mat getMat(){
                return this->frame->clone();
            }

            std::vector<cv::Scalar> vcolor = {
                    cv::Scalar(180, 77, 31),      //1
                    cv::Scalar(14, 127, 255),     //2  
                    cv::Scalar(194, 160, 194),    //3  
                    cv::Scalar(28, 27, 214),      //4  
                    cv::Scalar(94, 67, 189),      //5  
                    cv::Scalar(75, 56, 14),       //6  
                    cv::Scalar(194, 77, 227),     //7  
                    cv::Scalar(127, 127, 127),    //8  
                    cv::Scalar(22, 189, 188),     //9  
                    cv::Scalar(207, 190, 17)     //10 
                };

            Graph(){
                    init_MakeGraph();
                }  
            ~Graph(){
                if(this->frame != nullptr){
                    delete this->frame;
                }
            }
            

            void insert(std::vector<cv::Point2d>* vxy){
                this->vpvec.push_back(vxy);
            }


            void make(){
            	 if(frame != nullptr){    	 	
	                delete this->frame;
            	 }
                this->frame = new cv::Mat(map_size, CV_8UC3, map_backgraund_color);

                int i = 0;
                for(auto pvec : vpvec){
                    for(auto item : *pvec){
                        cv::circle(*this->frame, item + center,
                            this->circle_size, this->vcolor.at(i % 10), -1);
                        std::cout << this->num_of_window << "\n";
                    }
                    i++;
                }
            }


            const int circle_show(int ms){
                void make();

                cv::String winname = std::to_string(++this->num_of_window) + "th Graph";
                cv::resize(*this->frame, *this->frame, map_size/2);
                cv::flip(*this->frame, *this->frame, 0);
                cv::imshow(winname, *this->frame);

                cv::waitKeyEx(ms);

                return this->num_of_window;
            }// end function circle show
        
        }; // end class Graph

    }; // namespace Graph_

    Graph_::Graph graph;

}; // namespace chan

#endif
