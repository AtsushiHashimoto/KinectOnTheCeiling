#include "../macrodef.h"
#include "framemat.h"


DepthMat::DepthMat(int flag_v){
	flag_visualize=flag_v;
	name_framewindow=WINDOWNAME_DEPTH;
}

void DepthMat::aquireDescriptionFromSensor(IKinectSensor** pSensor,IFrameDescription** pDescription){
	HRESULT hResult;
	IDepthFrameSource* pDepthSource;

	hResult=(*pSensor)->get_DepthFrameSource(&pDepthSource);
	if(FAILED(hResult)){
		throw "Error : IKinectSensor::get_DepthFrameSource()";
	}
	hResult=pDepthSource->get_FrameDescription(pDescription);
	if(FAILED(hResult)){
		throw "Error : IDepthFrameSource::get_FrameDescription()";
	}
}

void DepthMat::setupMatWindowProperty(IFrameDescription* pDescription){
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

void DepthMat::initializeMat(IKinectSensor** pSensor){
	IFrameDescription* pDescription;

	aquireDescriptionFromSensor(pSensor,&pDescription);
	setupMatWindowProperty(pDescription);
}

void DepthMat::aquireDepthBufferMat(IDepthFrame** pDepthFrame){
	HRESULT hResult;
	hResult=(*pDepthFrame)->AccessUnderlyingBuffer(&buffersize,reinterpret_cast<UINT16**>(&bufferMat.data));
	if(SUCCEEDED(hResult)){
		imageMat=bufferMat.clone();
		imageMat.convertTo(displayMat,CV_8U,256.0f/4096.0f,0.0f);
	}else{
		throw "Error : IDepthFrame::AccessUnderlyingBuffer()";
	}
}
