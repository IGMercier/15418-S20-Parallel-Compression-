#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <omp.h>

#include "config.hh"
#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"
#include "dequantization.h"

using namespace std;
using pixel_t = uint8_t;


int n, m;

void discreteCosTransform(vector<vector<int>> &, int, int);
void free_mat(float **);
void divideMatrix(vector<vector<int>> &, int, int);


inline int getOffset(int width, int i, int j) {
    return (i * width + j) * 3;
}


template <typename T>
vector<vector<T>> initializeMatrix(int rows, int cols) {
    return vector<vector<T>>(rows, vector<T>(cols));
}


void divideMatrix(vector<vector<int>> &grayContent, int n, int m) {
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
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


void compress(pixel_t *img, int num, int width, int height) {
    n = height;
    m = width;

    int add_rows = (PIXEL - (n % PIXEL) != PIXEL ? PIXEL - (n % PIXEL) : 0);
    int add_columns = (PIXEL - (m % PIXEL) != PIXEL ? PIXEL - (m % PIXEL) : 0) ;

    /* Initialize data structures */
    vector<vector<int>> grayContent = vector<vector<int>>(n + add_rows, vector<int>(m + add_columns));
    globalDCT = vector<vector<float>>(n + add_rows, vector<float>(m + add_columns));
    finalMatrixCompress = vector<vector<int>>(n + add_rows, vector<int>(m + add_columns));
    finalMatrixDecompress = vector<vector<int>>(n + add_rows, vector<int>(m + add_columns));

    // vector<vector<int>> grayContent = initializeMatrix(n + add_rows, m + add_columns);
    // globalDCT = initializeMatrix(n + add_rows, m + add_columns);
    // finalMatrixCompress = initializeMatrix(n + add_rows, m + add_columns);
    // finalMatrixDecompress = initializeMatrix(n + add_rows, m + add_columns);

#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
#endif
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            pixel_t *bgrPixel = img + getOffset(width, i, j);
            grayContent[i][j] = (bgrPixel[0] + bgrPixel[1] + bgrPixel[2]) / 3.f;
        }
    }

    // zero-padding extra rows
#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
#endif
    for (int j = 0; j < m; j++) {
        for (int i = n; i < n + add_rows; i++) {
            grayContent[i][j] = 0;
        }
    }

    // zero-padding extra columns
#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
#endif
    for (int i = 0; i < n; i++) {
        for (int j = m; j < m + add_columns; j++) {
            grayContent[i][j] = 0;
        }
    }

    n += add_rows;      // making number of rows a multiple of 8
    m += add_columns;   // making number of cols a multiple of 8

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

#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
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

    // free(grayContent);
}


int main(int argc, char **argv) {
    FILE *fp;
    fp = fopen("./info.txt","a+"); 
    omp_set_num_threads(NUM_THREADS);
    // TODO: Check if dir exist
    string img_dir = "../../images/";
    string save_dir = "./compressed_images/";
    string ext = ".jpg";

    string img_name = argv[1] + ext;
    string path = img_dir + img_name;
    cout << img_name << ", ";

#ifdef OMP
    cout << "OMP, ";
#endif
#ifdef SIMD
    cout << "SIMD, ";
#endif

    auto start = chrono::high_resolution_clock::now();
    int width, height, bpp;
    uint8_t *img = stbi_load(path.data(), &width, &height, &bpp, 3);
    compress(img, 0, width, height);
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
