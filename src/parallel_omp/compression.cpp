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


void discreteCosTransform(float **, float **, int, int, int, int);
void free_mat(float **p);
void divideMatrix(float **grayContent, int dimX, int dimY, int n, int m);


inline int getOffset(int width, int i, int j) {
    return (i * width + j) * 3;
}


void print_mat(float **m, int N, int M){
   for(int i = 0; i < N; i++)
   {
    for(int j = 0; j < M; j++)
    {
      cout<<m[i][j]<<" ";
    }
    cout<<endl;
   }
   cout<<endl;
}


void divideMatrix(float **grayContent, int dimX, int dimY, int n, int m) {
    if (grayContent == nullptr || globalDCT == nullptr) {
        cout << "Invalid argument" << endl;
    }

#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
    for (int i = 0; i < n; i += dimX) {
        for (int j = 0; j < m; j += dimY) {
            discreteCosTransform(globalDCT, grayContent, dimX, dimY, i, j);
        }
    }
}


void discreteCosTransform(float **globalDCT, float **grayImage,
                              int N, int M, int offsetX, int offsetY) {
    int u, v, x, y;
    float cos1, cos2, temp;
    // Useful constants.
    float term1 = M_PI / (float)N;
    float term2 = M_PI / (float)M;
    float one_by_root_2 = 1.0 / sqrt(2);
    float one_by_root_2N = 1.0 / sqrt(2 * N);

    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
            temp = 0.0;
            for (x = 0; x < N; x++) {
                for (y = 0; y < M; y++) {
                    cos1 = cos(term1 * (x + 0.5) * u);
                    cos2 = cos(term2 * (y + 0.5) * v);
                    temp += grayImage[x + offsetX][y + offsetY] * cos1 * cos2;
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
    float **grayContent = calloc_mat(n + add_rows, m + add_columns);
    globalDCT = calloc_mat(n + add_rows, m + add_columns);
    // globalDCT = vector<vector<float>>(n, vector<float>(m));
    finalMatrixCompress = vector<vector<int>>(n + add_rows, vector<int>(m + add_columns, 0));
    finalMatrixDecompress = vector<vector<int>>(n + add_rows, vector<int>(m + add_columns, 0));

    int dimX = PIXEL, dimY = PIXEL;

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
    divideMatrix(grayContent, dimX, dimY, n, m);
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
    divideMatrix(grayContent, dimX, dimY, n, m);
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

    free(grayContent);
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
