#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "quantization.h"

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


void idct(float **Matrix, float **DCTMatrix, int N, int M) {
    int i, j, u, v;
    float cos1, cos2, temp;
    // Useful constants.
    float term1 = M_PI / (float)N;
    float term2 = M_PI / (float)M;
    float one_by_root_2 = 1.0 / sqrt(2);
    float one_by_root_2N = 1.0 / sqrt(2 * N);

    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
            Matrix[u][v] = 1/4. * DCTMatrix[0][0];
            for (i = 1; i < N; i++) {
                Matrix[u][v] += 1/2. * DCTMatrix[i][0];
            }

            for (j = 1; j < M; j++) {
                Matrix[u][v] += 1/2. * DCTMatrix[0][j];
            }

            for (i = 1; i < N; i++) {
                for (j = 1; j < M; j++) {
                    // cos1 = cos(term1 * (u + 1./2.) * i);
                    // cos2 = cos(term2 * (v + 1./2.) * j);
                    cos1 = cos(term1 * (i + 1./2.) * u);
                    cos2 = cos(term2 * (j + 1./2.) * v);
                    Matrix[u][v] += DCTMatrix[i][j] * cos1 * cos2;
                }
            }

            Matrix[u][v] *= 2./((float)N)*2./((float)M);
        }
    }
}


void dequantizeBlock(int R,int C) {
    int temp, i, j;
    float **finalBlock = calloc_mat(WINDOW, WINDOW);
    float **newBlock = calloc_mat(pixel, pixel);
    for (i = 0; i < WINDOW; i++) {
        for (j = 0; j < WINDOW; j++) {
            // read quantized value from compressed vector
            temp = finalMatrixCompress[R][C].v[i * WINDOW + j];
            // dequantize value using `quantArr`
            newBlock[i][j] = (float)temp * quantArr[i][j];
        }
    }

    idct(finalBlock, newBlock, WINDOW, WINDOW);

    for (i = 0; i < WINDOW; i++) {
        for (j = 0; j < WINDOW; j++) {
            finalMatrixDecompress[R * WINDOW + i][C * WINDOW + j] = finalBlock[i][j];
        }
    }
    return;
}


void dequantize(int n,int m) {
    int i, j;
    // #pragma omp parallel for schedule(runtime)
    for (i=0; i<n/pixel; i++) {
        for (j=0; j<m/pixel; j++) {
            dequantizeBlock(i, j);
        }
    }

    return;
}
