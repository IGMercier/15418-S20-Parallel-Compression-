#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"
#include "dequantization.h"

#define SERIAL 1
#if !SERIAL
#define OMP
#endif

#define NUM_THREADS 8
#define NUM_CHANNELS 3

using namespace std;

using pixel_t = uint8_t;

int n, m;

void dct2(float **DCTMatrix, float **Matrix, int N, int M, int, int);
void write_mat(FILE *fp, float **testRes, int N, int M);
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
        // cout << "Thread: " << endl;
        for (int j = 0; j < m; j += dimY) {
            dct2(globalDCT, grayContent, dimX, dimY, i, j);
        }
    }
}


void dct2(float **globalDCT, float **grayMatrix, int N, int M, int offsetX, int offsetY) {
    int u, v, x, y;
    float cos1, cos2, temp;
    // NOTE: assume that N and M are same.
    float term1 = M_PI / (float)N;
    float term2 = M_PI / (float)M;
    float one_by_root_2 = 1.0 / sqrt(2);
    float one_by_root_2N = 1.0 / sqrt(2 * N);

    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
            for (x = 0; x < N; x++) {
                for (y = 0; y < M; y++) {
                    cos1 = cos(term1 * (x + 0.5) * u);
                    cos2 = cos(term2 * (y + 0.5) * v);
                    temp += grayMatrix[x + offsetX][y + offsetY] * cos1 * cos2;
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
            // #pragma omp atomic
            globalDCT[u + offsetX][v + offsetY] = temp;
        }
    }
}


void compress(pixel_t *img, int num, int width, int height) {
    n = height;
    m = width;

    int add_rows = (pixel - (n % pixel) != pixel ? pixel - (n % pixel) : 0);
    int add_columns = (pixel - (m % pixel) != pixel ? pixel - (m % pixel) : 0) ;
    float **grayContent = calloc_mat(n + add_rows, m + add_columns);
    globalDCT = calloc_mat(n + add_rows, m + add_columns);
    // globalDCT = vector<vector<float>>(n, vector<float>(m));

    int dimX = pixel, dimY = pixel;

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

    // setting extra rows to 0
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

    // setting extra columns to 0
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

    n = add_rows + n;  // making rows as a multiple of 8
    m = add_columns + m;  // making columns as a multiple of 8

    divideMatrix(grayContent, dimX, dimY, n, m);
    quantize(n,m);
    dequantize(n,m);

#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
#endif
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++)    
        {
          pixel_t pixelValue = finalMatrixDecompress[i][j];
          pixel_t *bgrPixel = img + getOffset(width, i, j);
          bgrPixel[0] = pixelValue;
          bgrPixel[1] = pixelValue;
          bgrPixel[2] = pixelValue;
        }
      }

    free(grayContent);
}


bool imagesAreIdentical(pixel_t *img1, pixel_t *img2, int width, int height, int bpp) {
    long error = 0;
    int size = height * width * bpp;
    for (int i = 0; i < size; i++) {
        if (img1[i] != img2[i]) {
            error += abs(img1[i] - img2[i]);
        }
    }

    return error == 0;
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
