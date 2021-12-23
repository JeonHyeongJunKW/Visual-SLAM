
#include "MakeGraph.h"


// program main
int main(void)
{
    std::vector<cv::Point_<double>> vec1 = {
        cv::Point(50, 10), cv::Point(152, 100), 
        cv::Point(156, 53), cv::Point(152, 100), 
        cv::Point(432, 10), cv::Point(15, 100)
    };
    std::vector<cv::Point_<double>> vec2 = {
        cv::Point(50, 101), cv::Point(100, 100),
        cv::Point(50, 10), cv::Point(162, 100), 
        cv::Point(50, 35), cv::Point(152, 61)
    };
    chan::graph.insert(&vec1);
    chan::graph.insert(&vec2);
    chan::graph.circle_show(0);
}
