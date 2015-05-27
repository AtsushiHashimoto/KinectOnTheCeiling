#include "../macrodef.h"
#include "framemat.h"


ColorMat::ColorMat(int flag_v){
	flag_visualize=flag_v;
	name_framewindow=WINDOWNAME_COLOR;
}

void ColorMat::aquireDescriptionFromSensor(IKinectSensor** pSensor,IFrameDescription** pDescription){
	HRESULT hResult;
	IColorFrameSource* pColorSource;

	hResult=(*pSensor)->get_ColorFrameSource(&pColorSource);
	if(FAILED(hResult)){
		throw "Error : IKinectSensor::get_ColorFrameSource()";
	}
	hResult=pColorSource->get_FrameDescription(pDescription);
	if(FAILED(hResult)){
		throw "Error : IColorFrameSource::get_FrameDescription()";
	}
}

void ColorMat::setupMatWindowProperty(IFrameDescription* pDescription){
	int width,height;
	pDescription->get_Width(&width);
	pDescription->get_Height(&height);
	buffersize=width*height*4*sizeof(unsigned char);
	bufferMat=cv::Mat(height,width,CV_8UC4);
	imageMat=cv::Mat(height,width,CV_8UC4);
	displayMat=cv::Mat(height/2,width/2,CV_8UC4);
}

void ColorMat::initializeMat(IKinectSensor** pSensor){
	IFrameDescription* pDescription;

	aquireDescriptionFromSensor(pSensor,&pDescription);
	setupMatWindowProperty(pDescription);
}

void ColorMat::aquireColorBufferMat(IColorFrame** pColorFrame){
	HRESULT hResult;
	hResult=(*pColorFrame)->CopyConvertedFrameDataToArray(buffersize,reinterpret_cast<BYTE*>(bufferMat.data),ColorImageFormat::ColorImageFormat_Bgra);
	if(SUCCEEDED(hResult)){
		imageMat=bufferMat.clone();
		cv::resize(bufferMat,displayMat,cv::Size(),0.5,0.5);
	}else{
		throw "Error : IColorFrame::CopyConvertedFrameDataToArray()";
	}
}