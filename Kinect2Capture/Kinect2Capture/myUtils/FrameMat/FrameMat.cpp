#include <iomanip>
#include <iostream>
#include "framemat.h"

void FrameMat::aquireImageFileProperties(boost::program_options::variables_map* values_options,std::string str_cmd_output){

	flag_output=values_options->count(str_cmd_output);
	std::string str_orgfilepath=values_options->count(str_cmd_output) ? (*values_options)[str_cmd_output].as<std::string>() : "";
	filepath.append(str_orgfilepath);

	if(flag_output && !(boost::filesystem::exists(filepath.parent_path()))){
		std::string str_errorstatus="Error : " + filepath.parent_path().string() + "doesn't exist";
		throw str_errorstatus;
	}
}

void FrameMat::saveFrame(int frame_number,int zero_num){
	if(flag_output){
		std::stringstream ss_imgname;
		ss_imgname << (filepath.parent_path() / filepath.stem()).string() << std::setw(zero_num) << std::setfill('0') << frame_number << filepath.extension().string();

		boost::filesystem::path filepath_after(ss_imgname.str());
		str_filename=filepath_after.filename().string();
		cv::imwrite(ss_imgname.str(),imageMat);

		std::cout << ss_imgname.str() << "...saved." << std::endl;
	}else{
		str_filename="";
	}
}