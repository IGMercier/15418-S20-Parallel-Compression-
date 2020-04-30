#include <sys/time.h>
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

#define SERIAL 0
#if !SERIAL
#define OMP
#endif

#define NUM_THREADS 8
#define NUM_CHANNELS 3

using namespace std;

using pixel_t = uint8_t;

int n, m;

void dct(float **DCTMatrix, float **Matrix, int N, int M);
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
    float **smallMatrix = calloc_mat(dimX, dimY);
    float **DCTMatrix = calloc_mat(dimX, dimY);

#if !SERIAL

#endif
    for (int i = 0; i < n; i += dimX) {
        for (int j = 0; j < m; j += dimY) {
#if !SERIAL
#ifdef OMP
            // #pragma omp parallel for
#endif
#endif
            for (int k = i; k < i + pixel && k < n; k++) {
                for (int l = j; l < j + pixel && l < m; l++) {
                    smallMatrix[k-i][l-j] = grayContent[k][l];
                }
            }

            dct(DCTMatrix, smallMatrix, dimX, dimY);

            for (int k = i; k < i + pixel && k < n ;k++) {
                for (int l=j; l<j+pixel && l<m ;l++) {
                    globalDCT[k][l] = DCTMatrix[k-i][l-j];
                }
            }
        }
    }

    // TODO: I think the memory is not being freed properly for double pointers
    delete[] smallMatrix;
    delete[] DCTMatrix;
}


void dct(float **DCTMatrix, float **Matrix, int N, int M) {
    int i, j, u, v;
    // for all pixels in the 8 x 8 block
// #ifdef OMP
//     #pragma omp parallel for schedule(runtime)
// cout << getenv(OMP_STACKSIZE) << endl;
// #endif
// #pragma omp parallel

// #if !SERIAL
// #ifdef OMP
// #pragma omp parallel
// #endif
// #endif
{
    float cos1, cos2, temp;
    // NOTE: assume that N and M are same.
    float term1 = M_PI / (float)N;
    float term2 = M_PI / (float)M;
    float one_by_root_2 = 1.0 / sqrt(2);
    float one_by_root_2N = 1.0 / sqrt(2 * N);

// #if !SERIAL
// #ifdef OMP
//     #pragma omp for
// #endif
// #endif
    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
            for (i = 0; i < N; i++) {
                for (j = 0; j < M; j++) {
                    cos1 = cos(term1 * (i + 0.5) * u);
                    cos2 = cos(term2 * (j + 0.5) * v);
                    temp += Matrix[i][j] * cos1 * cos2;
                }
            }

            temp *= one_by_root_2N;
            if (u > 0) {
                temp *= one_by_root_2;
            }

            if (v > 0) {
                temp *= one_by_root_2;
            }

            // TODO: Should be round the value?
            DCTMatrix[u][v] = temp;
        }
    }
}

}


void compress(pixel_t *img, int num, int width, int height) {
    n = height;
    m = width;

    int add_rows = (pixel - (n % pixel) != pixel ? pixel - (n % pixel) : 0);
    int add_columns = (pixel - (m % pixel) != pixel ? pixel - (m % pixel) : 0) ;
    float **grayContent = calloc_mat(n + add_rows, m + add_columns);
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
    fp=fopen("./info.txt","a+"); 
    omp_set_num_threads(NUM_THREADS);
    string str="../../images/";
    string ext=".jpg";

    string path = str + argv[1] + ext;
    cout << path << endl;

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

#if SERIAL
    stbi_write_jpg("sample_ser.jpg", width, height, bpp, img, width * bpp);
#else
    stbi_write_jpg("sample_par.jpg", width, height, bpp, img, width * bpp);
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
