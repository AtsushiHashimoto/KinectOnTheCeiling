//
// Created by takagi on 15/10/24.
//

#ifndef IMAGELABELING_OPENING_H
#define IMAGELABELING_OPENING_H

#define OPENING_ITERATION 1

#include <opencv2/opencv.hpp>

void opening(cv::Mat*, int*);
void expansion(cv::Mat*, int*);
void contraction(cv::Mat*, int*);
bool sameValue(int*, int*);

#endif //IMAGELABELING_OPENING_H
