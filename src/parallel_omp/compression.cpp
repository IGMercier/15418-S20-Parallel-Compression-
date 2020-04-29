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

// NOTE: Uncomment to run the serial implementation
// #define SERIAL
#define TIMER
#ifndef SERIAL
// #define SIMD
#define OMP
#endif


// #include "../../include/config.hh"
#include "dequantization.h"
#include "quantization.h"

#define NUM_THREADS 8
#define NUM_CHANNELS 3


using namespace std;

using pixel_t = uint8_t;

// TODO: Change the name of these variables to height and width. n and m
// does not make it clear.
int n, m;

// Useful forward references.
void dct(float **DCTMatrix, float **Matrix, int N, int M);
void write_mat(FILE *fp, float **testRes, int N, int M);
void free_mat(float **p);
void divideMatrix(float **grayContent, int dimX, int dimY, int n, int m);


inline int getOffset(int width, int i, int j) {
    return (i * width + j) * NUM_CHANNELS;
}


void print_mat(float **m, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cout<<m[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}


void divideMatrix(float **grayContent, int dimX, int dimY, int n, int m) {
    float **DCTMatrix = calloc_mat(dimX, dimY);
    float **smallMatrix = calloc_mat(n,m);

#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (int i = 0; i < n; i += dimX) {
    for (int j = 0; j < m; j += dimY) {
        for (int k = i; k < i + pixel && k < n; k++) {
#ifdef SIMD
        #pragma omp simd
#endif
        for (int l = j; l < j + pixel && l < m; l++) {
            smallMatrix[k-i][l-j] = grayContent[k][l];
        }
        }

        dct(DCTMatrix, smallMatrix, dimX, dimY);

        for (int k=i; k<i+pixel && k<n; k++) {
#ifdef SIMD
        #pragma omp simd
#endif 
        for (int l=j; l<j+pixel && l<m ;l++) {
            globalDCT[k][l]=DCTMatrix[k-i][l-j];
        }
        }
    }
    }

    delete [] smallMatrix;
    delete [] DCTMatrix;
}


void dct(float **DCTMatrix, float **Matrix, int N, int M) {
    int i, j, u, v;

    for (u = 0; u < N; ++u) {
    for (v = 0; v < M; ++v) {
        DCTMatrix[u][v] = 0;
        for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            // TODO: Avoid using long lines.
            DCTMatrix[u][v] += Matrix[i][j] * cos(M_PI/((float)N)*(i+1./2.)*u)*cos(M_PI/((float)M)*(j+1./2.)*v);
        }
        }
    }
    }
}


void compress(pixel_t *img, int width, int height) {
    n = height;
    m = width;

    int add_rows = (pixel - (n % pixel) != pixel ? pixel - (n % pixel) : 0);
    int add_columns = (pixel - (m % pixel) != pixel ? pixel - (m % pixel) : 0);
    int dimX = pixel, dimY = pixel;

    float **grayContent = calloc_mat(n + add_rows, m + add_columns);

#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (int i = 0; i < n; i++) {
#ifdef SIMD
    #pragma omp simd
#endif
    for (int j = 0; j < m; j++) {
        pixel_t *bgrPixel = img + getOffset(width, i, j);
        grayContent[i][j] = (bgrPixel[0] + bgrPixel[1] + bgrPixel[2]) / 3;
    }
    }

    //setting extra rows to 0
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (int j = 0; j < m; j++) {
#ifdef SIMD    
    #pragma omp simd
#endif
    for (int i = n; i < n + add_rows; i++) {
        grayContent[i][j] = 0;
    }
    }
    // #pragma omp end parallel for simd

    //setting extra columns to 0
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (int i = 0; i < n; i++) {
#ifdef SIMD
        #pragma omp simd
#endif
        for (int j = m; j < m + add_columns; j++) {
            grayContent[i][j] = 0;
        }
    }
    // #pragma omp end parallel for simd

    n = add_rows + n;  // making rows as a multiple of 8
    m = add_columns + m;  // making columns as a multiple of 8

// NOTE: Change NO_TIMER to TIMER to get the timings of the functions.
#ifdef TIMER //NO_TIMER
    auto start = chrono::high_resolution_clock::now();
    divideMatrix(grayContent, dimX, dimY, n, m);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout << diff.count() << ", ";

    start = chrono::high_resolution_clock::now();
    quantize(n, m);
    end = chrono::high_resolution_clock::now();
    diff = end - start;
    cout << diff.count() << ", ";

    start = chrono::high_resolution_clock::now();
    dequantize(n, m);
    end = chrono::high_resolution_clock::now();
    diff = end - start;
    cout << diff.count()<< ", ";
#else
    divideMatrix(grayContent, dimX, dimY, n, m);
    quantize(n, m);
    dequantize(n, m);
#endif

#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (int i = 0; i < n; i++) {
#ifdef SIMD
    #pragma omp simd
#endif
    for (int j = 0; j < m; j++) {
        pixel_t pixelValue = finalMatrixDecompress[i][j];
        pixel_t *bgrPixel = img + getOffset(width, i, j);
        bgrPixel[0] = pixelValue;
        bgrPixel[1] = pixelValue;
        bgrPixel[2] = pixelValue;
    }
    }
    // #pragma omp end parallel for simd
    free(grayContent);
}


int main(int argc, char **argv) {
    FILE *fp;
    fp = fopen("./info.txt", "a+");
    // Set the number of threads
    omp_set_num_threads(NUM_THREADS);

    string image_dir = "../../images/";
    string image_ext = ".jpg";
    string path = image_dir + argv[1] + image_ext;
#ifdef OMP
    cout << "OMP, ";
#endif
#ifdef SIMD
    cout << "SIMD, ";
#endif
#ifdef SERIAL
    cout << "Serial -> ";
#else
    cout << "Parallel -> ";
#endif

    cout << "[ Image: " << argv[1] <<", ";
    // Start time
    auto start = chrono::high_resolution_clock::now();
    // Read and compress the image
    int width, height, bpp;
    pixel_t *img = stbi_load(path.data(), &width, &height, &bpp, NUM_CHANNELS);
    cout << "Width: " << width << ", Height: " << height << ", ";
    compress(img, width, height);
    stbi_image_free(img);
    // End time
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout << " t_compression: " << diff.count() << "]" << endl;
    fprintf(fp, "%f ", diff.count());
    fclose(fp);
    return 0;
}
