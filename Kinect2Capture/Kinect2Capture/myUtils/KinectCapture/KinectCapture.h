#ifndef CMDKINECTSENSOR_H
#define CMDKINECTSENSOR_H

#include <Kinect.h>
#include <memory>

class KinectCapture{

	IMultiSourceFrameReader* pMultiFrameReader;
	IKinectSensor* pSensor;

	bool flag_begin_aquireframe;		//フレーム取得が1度でも成功しているればtrue

public:
	void aquireKinectSensor();
	void aquireMultiSourceFrameReader();
	void initializeKinectCapture();

	IKinectSensor* getpSensor(){
		return pSensor;
	}
	IMultiSourceFrameReader* getpMultiFrameReader(){
		return pMultiFrameReader;
	}
	void KinectCapture::aquireMultiFrame(IColorFrame** pColorFrame,IDepthFrame** pDepthFrame,IInfraredFrame** pInfraredFrame);
	bool isBegunAquireFrame(){
		return flag_begin_aquireframe;
	}
};

#endif