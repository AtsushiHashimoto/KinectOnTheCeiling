//
// Created by takagi on 15/10/24.
//

#ifndef IMAGELABELING_LABELING_H
#define IMAGELABELING_LABELING_H

#define NUM_OF_PARTS 32

#include "opencv2/opencv.hpp"

void labeling(cv::Mat*, int*);
void labelValue(cv::Mat*, int*, const int, const int);
int rgbDist(cv::Vec3b, int*);

#endif //IMAGELABELING_LABELING_H
