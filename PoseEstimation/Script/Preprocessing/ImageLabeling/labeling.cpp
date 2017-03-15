//
// Created by takagi on 15/10/24.
//

#include "labeling.h"

using namespace std;

void labeling(cv::Mat *colorImage, int *nLabels) {
    
    for (int i = 0; i < NUM_OF_PARTS + 1; i++)
        nLabels[i] = 0;

    for (int y = 0; y < colorImage->rows; y++) {
        for (int x = 0; x < colorImage->cols; x++) {
           labelValue(colorImage, nLabels, y, x);
        }
    }

    return;
}


void labelValue(cv::Mat *colorImage, int *nLabels, const int y, const int x){

    int partLabels[NUM_OF_PARTS+1][3] = {{1,0,0}, {0,1,0}, {4,0,0}, {2,0,1}, {2,4,0}, {3,4,3}, {4,4,3}, {2,4,2}, {3,3,3}, {1,2,0}, {0,3,1},
                  {4,4,0}, {4,3,0}, {0,4,4}, {0,3,4}, {2,1,0}, {0,1,2}, {4,1,4}, {1,4,4}, {4,1,0}, {0,1,4}, {2,1,4},
                  {2,1,1}, {1,2,4}, {4,1,1}, {1,0,1}, {1,0,2}, {4,2,2}, {1,4,1}, {3,2,1}, {1,1,0}, {0,0,0}, {4,4,4}};

    for (int i = 0; i < NUM_OF_PARTS+1; i++) {
        for (int j = 0; j < 3; j++) {
            partLabels[i][j] = max(0, partLabels[i][j] * 64 - 1);
        }
    }

    cv::Vec3b rgbValue = colorImage->at<cv::Vec3b>(y, x);
    int minDist, p, d;
    minDist = INT_MAX;
    p = 0;
    for (int i = 0; i < NUM_OF_PARTS+1; i++){
        d = rgbDist(rgbValue, partLabels[i]);
        if (minDist > d){
            minDist = d;
            p = i;
        }
    }


    rgbValue = cv::Vec3b((unsigned char)partLabels[p][2], (unsigned char)partLabels[p][1], (unsigned char)partLabels[p][0]);

    nLabels[p]++;

    colorImage->at<cv::Vec3b>(y, x) = rgbValue;

    return;
}


int rgbDist(cv::Vec3b rgbValue, int *partLabel){
    int d = 0;
    unsigned int val1, val2;
    for (int j = 0; j < 3; j++) {
        val1 = (unsigned int)rgbValue.val[j];
        val2 = (unsigned int)partLabel[2-j];
        d += (val1 - val2) * (val1 - val2);
    }

    return d;
}

