#include "LocalMapping.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Node/NodeHandler.h"
using namespace std;
using namespace cv;

void LocalMapping(NodeHandler &nodehandler)
{
    while(true)
    {
        if(nodehandler.v_newKeyFrame.size()>0)
        {
            // cout<<nodehandler.v_newKeyFrame.size()<<"찾았다요놈 : "<<nodehandler.v_newKeyFrame[0]->CurrentFrame<<endl;
            nodehandler.m_sharedlock.lock();
            NewKeyFrameSet* newFrameSet =nodehandler.v_newKeyFrame[0];
            float* R_t_scale;//13개의 파라미터가 순서대로 나옵니다.
            newFrameSet->CurrentFrame->Get_Rt(R_t_scale);
            //R_t_scale~~~~~~이걸로 비주얼라이제이션

            //
            nodehandler.AddNewKeyFrame(nodehandler.v_newKeyFrame[0]->CurrentFrame);
            nodehandler.v_newKeyFrame.erase(nodehandler.v_newKeyFrame.begin());

            nodehandler.m_sharedlock.unlock();
        }
    }
}
