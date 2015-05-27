#include <iostream>
#include <Kinect.h>
#include <boost/exception/exception.hpp>
#include "myUtils/macrodef.h"
#include "myUtils/CmdLineOptions/CmdLineOptions.h"
#include "myUtils/KinectCapture/KinectCapture.h"
#include "myUtils/FrameMat/FrameMat.h"
#include "myUtils/TimeStamp/TimeStamp.h"

//���������
template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease)
{
	if(pInterfaceToRelease!=nullptr){
		pInterfaceToRelease->Release();
		pInterfaceToRelease=nullptr;
	}
}



int main(int argc,char** argv){
	// �R�}���h���C�������擾
	CmdLineOptions cmdoptions(argc,argv);
	try{
		cmdoptions.opt_description.add_options()
			("help,h","�w���v��\��")
			("visualize,v","�擾�����摜��\��")
			("zero_num,z",boost::program_options::value<int>(),"�t�@�C���ۑ����̃t���[���ԍ���0���߂̐����w�肷��")
			("output_depth",boost::program_options::value<std::string>(),"�[�x�摜��ۑ�����f�B���N�g����t�@�C���̌`�����w�肷��(��:path/to/dir/basename.png)")
			("output_color",boost::program_options::value<std::string>(),"�����摜��ۑ�����f�B���N�g����t�@�C���̌`�����w�肷��")
			("output_infrared",boost::program_options::value<std::string>(),"�ԊO���摜��ۑ�����f�B���N�g����t�@�C���̌`�����w�肷��")
			("output_timestamp",boost::program_options::value<std::string>(),"�����̏���[�x�摜,�����摜���ꂼ��̃t�@�C������CSV�`���ŕۑ�����(��:path/to/dir/timestamp.csv)")
		;
		cmdoptions.aquireOptionsCmdline();
	}catch(std::exception &e){
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// �w���v�擾
	if(cmdoptions.values_options.count("help")){
		std::cout << cmdoptions.opt_description << std::endl;
		return EXIT_SUCCESS;
	}
	
	// �Z���T�[�擾,�E�B���h�E�ݒ�
	KinectCapture kinectcapture;
	IKinectSensor* pSensor;

	int flag_visualize=cmdoptions.values_options.count("visualize");

	ColorMat colormat(flag_visualize);
	DepthMat depthmat(flag_visualize);
	InfraredMat infraredmat(flag_visualize);

	try{
		kinectcapture.initializeKinectCapture();
		pSensor=kinectcapture.getpSensor();
		colormat.initializeMat(&pSensor);
		depthmat.initializeMat(&pSensor);
		infraredmat.initializeMat(&pSensor);
	}catch(char const *str){
		std::cerr << str << std::endl;
		return EXIT_FAILURE;
	}

	// �摜�ۑ��ݒ�
	int zero_num=cmdoptions.values_options.count("zero_num") ? cmdoptions.values_options["zero_num"].as<int>() : ZERO_NUM_DEFAULT;	//�t���[���ԍ��̌���(�f�t�H���g��8��)
	unsigned int frame_num=0;
	try{
		colormat.aquireImageFileProperties(&(cmdoptions.values_options),"output_color");
		depthmat.aquireImageFileProperties(&(cmdoptions.values_options),"output_depth");
		infraredmat.aquireImageFileProperties(&(cmdoptions.values_options),"output_infrared");
	}catch(char const *str){
		std::cerr << str << std::endl;
		return EXIT_FAILURE;
	}

	// csv�t�@�C���ǂݍ���
	TimeStamp timestamp(cmdoptions.values_options.count("output_timestamp"));
	try{
		if(timestamp.isSaved()){
			timestamp.openFileStream(cmdoptions.values_options["output_timestamp"].as<std::string>());
		}
	}catch(char const *str){
		std::cerr << str << std::endl;
		return EXIT_FAILURE;
	}

	// [���݂̉摜���擾���`��,�ۑ�]���J��Ԃ�(ESC�L�[�ŏI��)
	while(1){

		IColorFrame* pColorFrame=nullptr;
		IDepthFrame* pDepthFrame=nullptr;
		IInfraredFrame* pInfraredFrame=nullptr;

		try{
			kinectcapture.aquireMultiFrame(&pColorFrame,&pDepthFrame,&pInfraredFrame);
			colormat.aquireColorBufferMat(&pColorFrame);
			depthmat.aquireDepthBufferMat(&pDepthFrame);
			infraredmat.aquireInfraredBufferMat(&pInfraredFrame);
			timestamp.getCurrentTime();

			SafeRelease(pColorFrame);
			SafeRelease(pDepthFrame);
			SafeRelease(pInfraredFrame);

			if(flag_visualize){
				colormat.displayImage();
				depthmat.displayImage();
				infraredmat.displayImage();
			}

			// �摜,timestamp�ۑ�
			// �f�o�b�O��...50�t���[�����Ƃɕۑ�

#ifdef _DEBUG
			if(frame_num%50==0){
#endif
				colormat.saveFrame(frame_num,zero_num);
				depthmat.saveFrame(frame_num,zero_num);
				infraredmat.saveFrame(frame_num,zero_num);
				timestamp.saveCsv(&colormat,&depthmat,&infraredmat);
#ifdef _DEBUG
			}
#endif

			if(cv::waitKey(10)==VK_ESCAPE){
				break;
			}

			// �܂�AquireLatestFrame���������ĂȂ��ꍇ�̓t���[�����̃J�E���g�����Ȃ�
			if(kinectcapture.isBegunAquireFrame()){
				++frame_num;
			}
		}catch(char const *str){
			// ��O����...�ϐ���������ă��[�v�̍ŏ���
			SafeRelease(pColorFrame);
			SafeRelease(pDepthFrame);
			SafeRelease(pInfraredFrame);
			std::cerr << str << "(framenumber:" << frame_num << ")" << std::endl;
			continue;
		}
	}

	// �I��������
	if(pSensor){
		pSensor->Close();
	}
	SafeRelease(pSensor);
	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
