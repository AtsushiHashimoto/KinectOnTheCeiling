#include "../macrodef.h"
#include "FrameMat.h"

InfraredMat::InfraredMat(int flag_v){
	flag_visualize=flag_v;
	name_framewindow=WINDOWNAME_INFRARED;
}

void InfraredMat::aquireDescriptionFromSensor(IKinectSensor** pSensor,IFrameDescription** pDescription){
	HRESULT hResult;
	IInfraredFrameSource* pInfraredSource;

	hResult=(*pSensor)->get_InfraredFrameSource(&pInfraredSource);
	if(FAILED(hResult)){
		throw "Error : IKinectSensor::get_InfraredFrameSource()";
	}
	hResult=pInfraredSource->get_FrameDescription(pDescription);
	if(FAILED(hResult)){
		throw "Error : IInfraredFrameSource::get_FrameDescription()";
	}
}

void InfraredMat::setupMatWindowProperty(IFrameDescription* pDescription){
	int width,height;
	pDescription->get_Width(&width);
	pDescription->get_Height(&height);
	buffersize=width*height*sizeof(unsigned short);
	bufferMat=cv::Mat(height,width,CV_16UC1);
	imageMat=cv::Mat(height,width,CV_16UC1);
	displayMat=cv::Mat(height,width,CV_16UC1);

	if(flag_visualize){
		cv::namedWindow(name_framewindow);
	}
}

void InfraredMat::initializeMat(IKinectSensor** pSensor){
	IFrameDescription* pDescription_infrared;

	aquireDescriptionFromSensor(pSensor,&pDescription_infrared);
	setupMatWindowProperty(pDescription_infrared);
}

void InfraredMat::aquireInfraredBufferMat(IInfraredFrame** pInfraredFrame){
	HRESULT hResult;
	hResult=(*pInfraredFrame)->AccessUnderlyingBuffer(&buffersize,reinterpret_cast<UINT16**>(&bufferMat.data));
	if(SUCCEEDED(hResult)){
		imageMat=bufferMat.clone();
		displayMat=imageMat.clone();
	}else{
		throw "Error : IDepthFrame::AccessUnderlyingBuffer()";
	}
}
