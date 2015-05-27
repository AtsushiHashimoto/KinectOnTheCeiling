#ifndef CMDTIMESTAMP_H
#define CMDTIMESTAMP_H

#include <fstream>
#include <sstream>
#include <string>
#include <atlstr.h>

class ColorMat;
class DepthMat;
class InfraredMat;

class TimeStamp{

	int flag_output_timestamp;
	std::string str_filepath_csv;
	std::ofstream ofs_csvfile;
	SYSTEMTIME time_frame;

public:

	TimeStamp(int _flag_output_timestamp);
	int isSaved(){
		return flag_output_timestamp;
	}
	void openFileStream(std::string filepath);
	void getCurrentTime(){
		if(flag_output_timestamp){
			GetLocalTime(&time_frame);
		}
	}
	std::string getTimeString();
	void saveCsv(ColorMat* colormat,DepthMat* depthmat,InfraredMat* infraredmat);
};

#endif