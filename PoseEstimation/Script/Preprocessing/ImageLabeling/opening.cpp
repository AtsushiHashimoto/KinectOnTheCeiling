//
// Created by takagi on 15/10/24.
//

#include "opening.h"


void opening(cv::Mat *colorImage, int *nLabels){
    for (int i = 0; i < OPENING_ITERATION; i++) {
        contraction(colorImage, nLabels);
    }
    for (int i = 0; i < OPENING_ITERATION + 2; i++) {
        expansion(colorImage, nLabels);
    }

    return;
}


void expansion(cv::Mat *colorImage, int *nLabels){

    int nRows = colorImage->rows;
    int nCols = colorImage->cols;
    int (*imageValues)[3];
    int (*changeValues)[3];
    int ether[3] = {100, 100, 100};
    imageValues = new int[(nRows + 2) * (nCols + 2)][3];
    changeValues = new int[nRows * nCols][3];

    for (int y = 0; y < nRows + 2; y++)
        for (int x = 0; x < nCols + 2; x++)
            for (int j = 0; j < 3; j++)
                imageValues[y * (nCols + 2)  + x][j] = 255;

    for (int y = 0; y < nRows; y++){
        for (int x = 0; x < nCols; x++){
            for (int j = 0; j < 3; j++) {
                imageValues[(y + 1) * (nCols + 2)  + x + 1][j] = (unsigned int)colorImage->at<cv::Vec3b>(y, x).val[2-j];
            }
        }
    }

    for (int y = 0; y < nRows; y++){
        for (int x = 0; x < nCols; x++){
            if (sameValue(imageValues[(y + 1) * (nCols + 2)  + x + 1], ether)){
                for (int k = -1; k <= 1; k += 2) {
                    if (!sameValue(imageValues[(y + 1 + k) * (nCols + 2)  + x + 1], ether)){
                        for (int j = 0; j < 3; j++){
                            changeValues[y * nCols + x][j] = imageValues[(y + 1 + k) * (nCols + 2)  + x + 1][j];
                        }
                        break;
                    }
                    if (!sameValue(imageValues[(y + 1) * (nCols + 2)  + x + 1 + k], ether)){
                        for (int j = 0; j < 3; j++){
                            changeValues[y * nCols + x][j] = imageValues[(y + 1) * (nCols + 2)  + x + 1 + k][j];
                        }
                        break;
                    }
                    for (int j = 0; j < 3; j++){
                        changeValues[y * nCols + x][j] = imageValues[(y + 1) * (nCols + 2)  + x + 1][j];
                    }
                }
            } else {
                for (int j = 0; j < 3; j++) {
                    changeValues[y * nCols + x][j] = imageValues[(y + 1) * (nCols + 2)  + x + 1][j];
                }
            }
        }
    }

    for (int y = 0; y < nRows; y++){
        for (int x = 0; x < nCols; x++){
            colorImage->at<cv::Vec3b>(y, x) = cv::Vec3b((unsigned char)changeValues[y * nCols + x][2], (unsigned char)changeValues[y * nCols + x][1], (unsigned char)changeValues[y * nCols + x][0]);
        }
    }

    delete[] imageValues;
    delete[] changeValues;

    return;

}

void contraction(cv::Mat *colorImage, int *nLabels){

    int nRows = colorImage->rows;
    int nCols = colorImage->cols;
    int (*imageValues)[3];
    bool *innerPixels;
    imageValues = new int[(nRows + 2) * (nCols + 2)][3];
    innerPixels = new bool[nRows * nCols];

    for (int y = 0; y < nRows + 2; y++) {
        for (int x = 0; x < nCols + 2; x++) {
            for (int j = 0; j < 3; j++) {
                imageValues[y * (nCols + 2) + x][j] = 0;
            }
        }
    }

    for (int y = 0; y < nRows; y++) {
        for (int x = 0; x < nCols; x++) {
            for (int j = 0; j < 3; j++) {
                imageValues[(y + 1) * (nCols + 2) + x + 1][j] = (int)colorImage->at<cv::Vec3b>(y, x).val[2-j];
            }
        }
    }

    for (int y = 0; y < nRows; y++){
        for (int x = 0; x < nCols; x++){
            innerPixels[y * nCols + x] = true;
            for (int k = -1; k <= 1; k += 2) {
                if (!sameValue(imageValues[(y + 1 + k) * (nCols + 2) + x + 1], imageValues[(y + 1) * (nCols + 2) + x + 1])){
                    innerPixels[y * nCols + x] = false;
                    break;
                }
                if (!sameValue(imageValues[(y + 1) * (nCols + 2) + x + 1 + k], imageValues[(y + 1) * (nCols + 2) + x + 1])){
                    innerPixels[y * nCols + x] = false;
                    break;
                }
            }
        }
    }

    for (int y = 0; y < nRows; y++){
        for (int x = 0; x < nCols; x++){
            if (!innerPixels[y * nCols + x]) {
                colorImage->at<cv::Vec3b>(y, x) = cv::Vec3b((unsigned char)100, (unsigned char)100, (unsigned char)100);
            }
        }
    }
    delete[] imageValues;
    delete[] innerPixels;

    return;

}

bool sameValue(int rgbValue1[3], int rgbValue2[3]){
    bool same = true;
    for (int j = 0; j < 3; j++){
        if (rgbValue1[j] != rgbValue2[j]){
            same = false;
            break;
        }
    }
    return same;
}
