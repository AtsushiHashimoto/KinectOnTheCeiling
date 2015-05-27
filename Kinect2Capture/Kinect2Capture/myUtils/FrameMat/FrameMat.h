#ifndef CMDFRAMEMAT_H
#define CMDFRAMEMAT_H

#include <string>
#include <Kinect.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

class FrameMat{
	
protected:
	std::string str_filename;
	int flag_output,flag_visualize;
	boost::filesystem::path filepath;
	std::string name_framewindow;

	unsigned int buffersize;
	cv::Mat bufferMat,imageMat,displayMat;

public:
	void aquireImageFileProperties(boost::program_options::variables_map* values_options,std::string str_cmd_output);
	void saveFrame(int frame_number,int zero_num);
	
	void displayImage(){
		cv::imshow(name_framewindow,displayMat);
	}
	bool isSaved(){
		return flag_output;
	}
	std::string getFileName(){
		return str_filename;
	}
};


class ColorMat : public FrameMat{
	
public:
	ColorMat(int flag_v);
	void aquireDescriptionFromSensor(IKinectSensor** pSensor,IFrameDescription** pDescription);
	void setupMatWindowProperty(IFrameDescription* pDescription);
	void initializeMat(IKinectSensor** pSensor);

	void aquireColorBufferMat(IColorFrame** pColorFrame);
};


class DepthMat : public FrameMat{

public:
	DepthMat(int flag_v);
	void aquireDescriptionFromSensor(IKinectSensor** pSensor,IFrameDescription** pDescription);
	void setupMatWindowProperty(IFrameDescription* pDescription);
	void initializeMat(IKinectSensor** pSensor);

	void aquireDepthBufferMat(IDepthFrame** pDepthFrame);
};


class InfraredMat : public FrameMat{

public:
	InfraredMat(int flag_v);
	void aquireDescriptionFromSensor(IKinectSensor** pSensor,IFrameDescription** pDescription);
	void setupMatWindowProperty(IFrameDescription* pDescription);
	void initializeMat(IKinectSensor** pSensor);

	void aquireInfraredBufferMat(IInfraredFrame** pInfraredFrame);
};

#endif