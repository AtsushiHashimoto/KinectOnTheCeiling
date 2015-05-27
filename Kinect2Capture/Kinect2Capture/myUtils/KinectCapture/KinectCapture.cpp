#include "KinectCapture.h"

void KinectCapture::aquireKinectSensor(){
	HRESULT hResult;

	hResult=GetDefaultKinectSensor(&pSensor);
	if(FAILED(hResult)){
		throw "Error : GetDefaultKinectSensor";
	}
}

void KinectCapture::aquireMultiSourceFrameReader(){
	HRESULT hResult;

	hResult=pSensor->Open();
	if(FAILED(hResult)){
		throw "Error : IKinectSensor::Open()";
	}

	hResult=pSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color|FrameSourceTypes_Depth|FrameSourceTypes_Infrared,&pMultiFrameReader);
	if(FAILED(hResult)){
		throw "Error : IKinectSensor::OpenMultiSourceFrameReader()";
	}
}

void KinectCapture::initializeKinectCapture(){
	flag_begin_aquireframe=false;

	aquireKinectSensor();
	aquireMultiSourceFrameReader();
}


void KinectCapture::aquireMultiFrame(IColorFrame** pColorFrame,IDepthFrame** pDepthFrame,IInfraredFrame** pInfraredFrame){
	HRESULT hResult=S_OK;
	IMultiSourceFrame* pMultiFrame=nullptr;
	hResult=pMultiFrameReader->AcquireLatestFrame(&pMultiFrame);
	if(FAILED(hResult)){
		throw "Error : IMultiFrameReader::AcquireLatestFrame()";
	}else{
		//IMultiSourceFrameReader::AquireLatestFrameが初めて成功したときにflagをtrueにする→フレームのカウントが可能になる
		if(!flag_begin_aquireframe){
			flag_begin_aquireframe=true;
		}
	}

	IColorFrameReference* pColorFrameReference=nullptr;
	IDepthFrameReference* pDepthFrameReference=nullptr;
	IInfraredFrameReference* pInfraredFrameReference=nullptr;
	hResult=pMultiFrame->get_ColorFrameReference(&pColorFrameReference);
	if(FAILED(hResult)){
		throw "Error : IMultiSourceFrame::get_ColorFrameReference()";
	}
	hResult=pMultiFrame->get_DepthFrameReference(&pDepthFrameReference);
	if(FAILED(hResult)){
		throw "Error : IMultiSourceFrame::get_DepthFrameReference()";
	}
	hResult=pMultiFrame->get_InfraredFrameReference(&pInfraredFrameReference);
	if(FAILED(hResult)){
		throw "Error : IMultiSourceFrame::get_InfraredFrameReference()";
	}
	hResult=pColorFrameReference->AcquireFrame(pColorFrame);
	if(FAILED(hResult)){
		throw "Error : IColorFrameReference::AcquireFrame()";
	}
	hResult=pDepthFrameReference->AcquireFrame(pDepthFrame);
	if(FAILED(hResult)){
		throw "Error : IDepthFrameReference::AcquireFrame()";
	}
	hResult=pInfraredFrameReference->AcquireFrame(pInfraredFrame);
	if(FAILED(hResult)){
		throw "Error : IInfraredFrameReference::AcquireFrame()";
	}

}
