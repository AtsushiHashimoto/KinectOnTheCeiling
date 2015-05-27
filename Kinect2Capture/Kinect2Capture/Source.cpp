#include <iostream>
#include <Kinect.h>
#include <boost/exception/exception.hpp>
#include "myUtils/macrodef.h"
#include "myUtils/CmdLineOptions/CmdLineOptions.h"
#include "myUtils/KinectCapture/KinectCapture.h"
#include "myUtils/FrameMat/FrameMat.h"
#include "myUtils/TimeStamp/TimeStamp.h"

//メモリ解放
template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease)
{
	if(pInterfaceToRelease!=nullptr){
		pInterfaceToRelease->Release();
		pInterfaceToRelease=nullptr;
	}
}



int main(int argc,char** argv){
	// コマンドライン引数取得
	CmdLineOptions cmdoptions(argc,argv);
	try{
		cmdoptions.opt_description.add_options()
			("help,h","ヘルプを表示")
			("visualize,v","取得した画像を表示")
			("zero_num,z",boost::program_options::value<int>(),"ファイル保存時のフレーム番号の0埋めの数を指定する")
			("output_depth",boost::program_options::value<std::string>(),"深度画像を保存するディレクトリやファイルの形式を指定する(例:path/to/dir/basename.png)")
			("output_color",boost::program_options::value<std::string>(),"可視光画像を保存するディレクトリやファイルの形式を指定する")
			("output_infrared",boost::program_options::value<std::string>(),"赤外線画像を保存するディレクトリやファイルの形式を指定する")
			("output_timestamp",boost::program_options::value<std::string>(),"時刻の情報や深度画像,可視光画像それぞれのファイル名をCSV形式で保存する(例:path/to/dir/timestamp.csv)")
		;
		cmdoptions.aquireOptionsCmdline();
	}catch(std::exception &e){
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// ヘルプ取得
	if(cmdoptions.values_options.count("help")){
		std::cout << cmdoptions.opt_description << std::endl;
		return EXIT_SUCCESS;
	}
	
	// センサー取得,ウィンドウ設定
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

	// 画像保存設定
	int zero_num=cmdoptions.values_options.count("zero_num") ? cmdoptions.values_options["zero_num"].as<int>() : ZERO_NUM_DEFAULT;	//フレーム番号の桁数(デフォルトは8桁)
	unsigned int frame_num=0;
	try{
		colormat.aquireImageFileProperties(&(cmdoptions.values_options),"output_color");
		depthmat.aquireImageFileProperties(&(cmdoptions.values_options),"output_depth");
		infraredmat.aquireImageFileProperties(&(cmdoptions.values_options),"output_infrared");
	}catch(char const *str){
		std::cerr << str << std::endl;
		return EXIT_FAILURE;
	}

	// csvファイル読み込み
	TimeStamp timestamp(cmdoptions.values_options.count("output_timestamp"));
	try{
		if(timestamp.isSaved()){
			timestamp.openFileStream(cmdoptions.values_options["output_timestamp"].as<std::string>());
		}
	}catch(char const *str){
		std::cerr << str << std::endl;
		return EXIT_FAILURE;
	}

	// [現在の画像を取得→描画,保存]を繰り返す(ESCキーで終了)
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

			// 画像,timestamp保存
			// デバッグ時...50フレームごとに保存

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

			// まだAquireLatestFrameが成功してない場合はフレーム数のカウントをしない
			if(kinectcapture.isBegunAquireFrame()){
				++frame_num;
			}
		}catch(char const *str){
			// 例外処理...変数を解放してループの最初へ
			SafeRelease(pColorFrame);
			SafeRelease(pDepthFrame);
			SafeRelease(pInfraredFrame);
			std::cerr << str << "(framenumber:" << frame_num << ")" << std::endl;
			continue;
		}
	}

	// 終了時処理
	if(pSensor){
		pSensor->Close();
	}
	SafeRelease(pSensor);
	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
