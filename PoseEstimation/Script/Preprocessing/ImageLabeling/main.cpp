// main.cpp
#include "main.h"

using namespace std;
namespace fs = boost::filesystem;

int main(int argc, char* argv[]) {

    string imgPath, target_ID, inFile, outFile;
    string figures[2] = {"female", "male"};
    cv::Mat colorImage;
    int *nLabels;
    int n_fig = 500000;
    bool reg_file, png_file, not_yet, not_depth;

    if (argc == 2){
        imgPath = argv[1];
    } else if (argc == 1) {
        imgPath = "/Users/mideal/Public/takagi/PoseEstimation/Data/Main/BodyPartClassification/SyntheticImages/";
    } else {
        cerr << "Wrong number of command line parameters." << endl;
    }

    for (int j = 0; j < 2; j++) {
        cout << figures[j] << endl;
        for (int i = 0; i < n_fig; i++) {
            #pragma omp parallel for
            for (int k = 0; k < 64; k++) {
                stringstream ss;
                ss << setfill('0') << setw(5) << i;
                target_ID = ss.str() + "_" + to_string(k);
                inFile = imgPath + figures[j] + "_for_proposed/" + target_ID + ".png";
                outFile = imgPath + figures[j] + "_for_proposed/" + target_ID + "_labeled.png";
                fs::path pout(outFile), pin(inFile);
                cout << inFile << endl;
                if (!fs::exists(pout) && fs::exists(pin)) {
                    cout << figures[j] + "_for_proposed/" + target_ID << endl;
                    colorImage = cv::imread(inFile, 1);

                    nLabels = new int[NUM_OF_PARTS+1];

                    labeling(&colorImage, nLabels);
                    opening(&colorImage, nLabels);
                    cv::imwrite(outFile, colorImage);
                    colorImage.release();

                    delete[] nLabels;

        //            for (int i = 0; i < NUM_OF_PARTS+1; i++)
        //                cout << "Part[" << i+1 << "] :" << nLabels[i] << endl;
                }
            }
        }
    }


    return 0;
}

