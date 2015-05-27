#include "CmdLineOptions.h"

CmdLineOptions::CmdLineOptions(int argc_cmd,char* argv_cmd[]){
	argc=argc_cmd;
	argv=argv_cmd;
}

void CmdLineOptions::aquireOptionsCmdline(){
	boost::program_options::store(boost::program_options::parse_command_line(argc,argv,opt_description),values_options);
	boost::program_options::notify(values_options);
}