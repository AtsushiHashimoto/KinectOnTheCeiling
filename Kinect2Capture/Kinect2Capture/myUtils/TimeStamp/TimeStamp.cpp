#include <iomanip>
#include "../macrodef.h"
#include "../framemat/framemat.h"
#include "timestamp.h"


inline void printZeroFillNum(std::ostream& out,int num,int zero_num=ZERO_NUM_TIME){
	out << std::setw(zero_num) << std::setfill('0') << num;
}

TimeStamp::TimeStamp(int _flag_output_timestamp){
	flag_output_timestamp=_flag_output_timestamp;
}


void TimeStamp::openFileStream(std::string filepath){
	str_filepath_csv=filepath;
	ofs_csvfile.open(str_filepath_csv,std::ios::trunc);
	if(!ofs_csvfile){
		throw "Error : std::ofstream::open()";
	}
}

std::string TimeStamp::getTimeString(){
	std::stringstream ss_time;
	ss_time << time_frame.wYear;
	printZeroFillNum(ss_time,time_frame.wMonth);
	printZeroFillNum(ss_time,time_frame.wDay);
	ss_time << "_";
	printZeroFillNum(ss_time,time_frame.wHour);
	printZeroFillNum(ss_time,time_frame.wMinute);
	printZeroFillNum(ss_time,time_frame.wSecond);
	ss_time << "_";
	printZeroFillNum(ss_time,time_frame.wMilliseconds,ZERO_NUM_MSECOND);
	return ss_time.str();
}


void TimeStamp::saveCsv(ColorMat* colormat,DepthMat* depthmat,InfraredMat* infraredmat){

	if(flag_output_timestamp){
		ofs_csvfile << getTimeString();
		if(colormat->isSaved()){
			ofs_csvfile << "," << colormat->getFileName();
		}
		if(depthmat->isSaved()){
			ofs_csvfile << "," << depthmat->getFileName();
		}
		if(infraredmat->isSaved()){
			ofs_csvfile << "," << infraredmat->getFileName();
		}
		ofs_csvfile << std::endl;
	}
}
