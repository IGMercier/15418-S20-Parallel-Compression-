#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cuda.h>

#include "../include/config_cuda.hh"
#include "../../../include/stb_image.h"
#include "../../../include/stb_image_write.h"
#include "dequantization.hh"

using namespace std;
using pixel_t = uint8_t;


int n, m;

void discreteCosTransform(vector<vector<int>> &, int, int);
void free_mat(float **);
void divideMatrix(vector<vector<int>> &, int, int);


inline int getOffset(int width, int i, int j) {
    return (i * width + j) * 3;
}


int **initializeIntPointerMatrix(int rows, int cols) {
    int **ptr = (int **) new int*[rows];
    for (int i = 0; i < rows; i++) {
        ptr[i] = new int[cols];
    }
    return ptr;
}


float **initializeFloatPointerMatrix(int rows, int cols) {
    float **ptr = (float **) new float*[rows];
    for (int i = 0; i < rows; i++) {
        ptr[i] = (float *) new float[cols];
    }
    return ptr;
}


void freeIntMatrix(int **ptr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        delete[] ptr[i];
    }
    delete[] ptr;
}


void freeFloatMatrix(float **ptr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        delete[] ptr[i];
    }
    delete[] ptr;
}


void divideMatrix(vector<vector<int>> &grayContent, int n, int m) {
    for (int i = 0; i < n; i += WINDOW_X) {
        for (int j = 0; j < m; j += WINDOW_Y) {
            discreteCosTransform(grayContent, i, j);
        }
    }
}


void discreteCosTransform(vector<vector<int>> &grayContent, int offsetX, int offsetY) {
    int u, v, x, y;
    float cos1, cos2, temp;
    // Useful constants.
    float term1 = M_PI / (float)WINDOW_X;
    float term2 = M_PI / (float)WINDOW_Y;
    float one_by_root_2 = 1.0 / sqrt(2);
    float one_by_root_2N = 1.0 / sqrt(2 * WINDOW_X);

    for (u = 0; u < WINDOW_X; ++u) {
        for (v = 0; v < WINDOW_Y; ++v) {
            temp = 0.0;
            for (x = 0; x < WINDOW_X; x++) {
                for (y = 0; y < WINDOW_Y; y++) {
                    cos1 = cos(term1 * (x + 0.5) * u);
                    cos2 = cos(term2 * (y + 0.5) * v);
                    temp += grayContent[x + offsetX][y + offsetY] * cos1 * cos2;
                }
            }

            temp *= one_by_root_2N;
            if (u > 0) {
                temp *= one_by_root_2;
            }

            if (v > 0) {
                temp *= one_by_root_2;
            }

            // TODO: ensure that u + offset < i + pixel and < n
            globalDCT[u + offsetX][v + offsetY] = temp;
        }
    }
}


void compress(pixel_t *const img, int width, int height) {
    n = height;
    m = width;

    int add_rows = (PIXEL - (n % PIXEL) != PIXEL ? PIXEL - (n % PIXEL) : 0);
    int add_columns = (PIXEL - (m % PIXEL) != PIXEL ? PIXEL - (m % PIXEL) : 0) ;

    // padded dimensions to make multiples of patch size
    int _height = n + add_rows;
    int _width = m + add_columns;

    // Initialize data structures
    vector<vector<int>> grayContent = initializeIntMatrix(_height, _width);
    globalDCT = initializeFloatMatrix(_height, _width);
    finalMatrixCompress = initializeIntMatrix(_height, _width);
    finalMatrixDecompress = initializeIntMatrix(_height, _width);

    for (int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            pixel_t *bgrPixel = img + getOffset(width, i, j);
            grayContent[i][j] = (bgrPixel[0] + bgrPixel[1] + bgrPixel[2]) / 3.f;
        }
    }

    // zero-padding extra rows
    for (int j = 0; j < m; j++) {
        for (int i = n; i < n + add_rows; i++) {
            grayContent[i][j] = 0;
        }
    }

    // zero-padding extra columns
    for (int i = 0; i < n; i++) {
        for (int j = m; j < m + add_columns; j++) {
            grayContent[i][j] = 0;
        }
    }

    n = _height;      // making number of rows a multiple of 8
    m = _width;   // making number of cols a multiple of 8

#ifdef TIMER // NO_TIMER
    auto start = chrono::high_resolution_clock::now();
    divideMatrix(grayContent, n, m);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout << "DCT: " << diff.count() << ", ";

    start = chrono::high_resolution_clock::now();
    quantize(n, m);
    end = chrono::high_resolution_clock::now();
    diff = end - start;
    cout << "Quant: " << diff.count() << ", ";

    start = chrono::high_resolution_clock::now();
    dequantize(n, m);
    end = chrono::high_resolution_clock::now();
    diff = end - start;
    cout << "Dequant: " << diff.count() << ", ";

    start = chrono::high_resolution_clock::now();
    invDct(n, m);
    end = chrono::high_resolution_clock::now();
    diff = end - start;
    cout << "IDCT: " << diff.count() << ", ";
#else
    divideMatrix(grayContent, n, m);
    quantize(n, m);
    dequantize(n, m);
    invDct(n, m);
#endif

    for (int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            pixel_t pixelValue = finalMatrixDecompress[i][j];
            pixel_t *bgrPixel = img + getOffset(width, i, j);
            bgrPixel[0] = pixelValue;
            bgrPixel[1] = pixelValue;
            bgrPixel[2] = pixelValue;
        }
    }
}



void cudaSetup(pixel_t *img, int width, int height) {
    grayContent = initializeIntPointerMatrix(height, width);
    finalMatrixCompress = initializeIntPointerMatrix(height, width);
    finalMatrixDecompress = initializeIntPointerMatrix(height, width);
    globalDCT = initializeFloatPointerMatrix(height, width);

    size_t num = NUM_CHANNELS * width * height;
    cudaMalloc(&cudaImg, sizeof(pixel_t) * num);
    cudaMalloc(&cudaGrayContent, sizeof(int) * num);
    cudaMalloc(&cudaGlobalDCT, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaFinalMatrixCompress, sizeof(int) * 3 * numberOfCircles);
    cudaMalloc(&cudaFinalMatrixDecompress, sizeof(int) * numberOfCircles);
    
}


int main(int argc, char **argv) {
    FILE *fp;
    fp = fopen("./info.txt","a+"); 
    // TODO: Check if dir exist
    string img_dir = "../../images/";
    string save_dir = "./compressed_images/";
    string ext = ".jpg";

    string img_name = argv[1] + ext;
    string path = img_dir + img_name;
    cout << img_name << ", ";

#ifdef CUDA
    cout << "CUDA, ";
#endif

    auto start = chrono::high_resolution_clock::now();
    int width, height, bpp;
    pixel_t *const img = stbi_load(path.data(), &width, &height, &bpp, 3);
    cudaSetup()

    compress(img, width, height);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_parallel = end - start;
    cout << "Width: " << width << ", ";
    cout << "Height: " << height << ", ";

#if SERIAL
    string save_img = save_dir + "ser_" + img_name;
    stbi_write_jpg(save_img.data(), width, height, bpp, img, width * bpp);
#else
    string save_img = save_dir + "par_" + img_name;
    stbi_write_jpg(save_img.data(), width, height, bpp, img, width * bpp);
#endif
    stbi_image_free(img);

#if SERIAL
    cout << "Serial -> ";
#else
    cout << "Parallel -> ";
#endif   
    cout << diff_parallel.count() << endl;

    fprintf(fp,"%f ",(float)diff_parallel.count());
    fclose(fp);
    return 0;
}
