#ifndef CMDLINEOPTIONS_H
#define CMDLINEOPTIONS_H

#include <boost/program_options.hpp>

class CmdLineOptions{
	int argc;
	char** argv;

public:
	boost::program_options::options_description opt_description;
	boost::program_options::variables_map values_options;

	CmdLineOptions(int argc_cmd,char* argv_cmd[]);
	void aquireOptionsCmdline();
};

#endif