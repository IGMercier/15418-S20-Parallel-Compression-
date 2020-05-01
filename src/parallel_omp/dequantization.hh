#ifndef DEQUANTIZATION_HH
#define DEQUANTIZATION_HH

#include <vector>
#include <cmath>

#include "../../include/config.hh"
#include "quantization.hh"

using namespace std;


// prototypes
void idct(float **Matrix, float **DCTMatrix, int N, int M);
float **calloc_mat(int dimX, int dimY);


float **calloc_mat(int dimX, int dimY) {
    float **m = (float**) calloc(dimX, sizeof(float*));
    float *p = (float *) calloc(dimX*dimY, sizeof(float));
    int i;

    for (i=0; i <dimX;i++) {
        m[i] = &p[i*dimY];
    }

    return m;
}


void free_mat(float **m){
    free(m[0]);
    free(m);
}


void invDiscreteCosTransform(int R, int C) {
    int x, y, u, v;
    float cos1, cos2, temp;
    // Useful constants.
    float term1 = M_PI / (float)WINDOW_X;
    float term2 = M_PI / (float)WINDOW_Y;
    float term3 = 2. / (float)WINDOW_X;
    float term4 = 2. / (float)WINDOW_Y;

    for (u = 0; u < WINDOW_X; ++u) {
        for (v = 0; v < WINDOW_Y; ++v) {
            temp = 1/4. * (float)finalMatrixCompress[R + 0][C + 0];
            for (x = 1; x < WINDOW_X; x++) {
                temp += 1/2. * (float)finalMatrixCompress[R + x][C + 0];
            }

            for (y = 1; y < WINDOW_Y; y++) {
                temp += 1/2. * (float)finalMatrixCompress[R + 0][C + y];
            }

            for (x = 1; x < WINDOW_X; x++) {
                for (y = 1; y < WINDOW_Y; y++) {
                    cos1 = cos(term1 * (x + 0.5) * u);
                    cos2 = cos(term2 * (y + 0.5) * v);
                    temp += (float)finalMatrixCompress[R + x][C + y] * cos1 * cos2;
                }
            }

            finalMatrixDecompress[u + R][v + C] = temp * term3 * term4;
        }
    }
}


void invDct(int height, int width) {
#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
#endif
    for (int i = 0; i < height; i += WINDOW_X) {
        for (int j = 0; j < width; j += WINDOW_Y) {
            invDiscreteCosTransform(i, j);
        }
    }
}


void dequantizeBlock(int R, int C) {
    int i, j;
    for (i = 0; i < WINDOW_X; i++) {
        for (j = 0; j < WINDOW_Y; j++) {
            // read and dequantize the quantized value from compressed vector
            finalMatrixCompress[R + i][C + j] *= quantArr[i][j];
        }
    }
}


void dequantize(int height, int width) {
#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
#endif
    for (int i = 0; i < height; i += WINDOW_X) {
        for (int j = 0; j < width; j += WINDOW_Y) {
            dequantizeBlock(i, j);
        }
    }
}

#endif
