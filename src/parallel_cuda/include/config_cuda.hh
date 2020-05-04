#pragma once

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define PIXEL 8
#define WINDOW_X 8
#define WINDOW_Y 8

#define SERIAL 0
#if !SERIAL
#define CUDA
#define OMP
#endif

/* If CUDA is defined */
#ifdef CUDA

#define BLK_WIDTH 32
#define BLK_HEIGHT 32
#define BLOCKSIZE (BLK_HEIGHT * BLK_WIDTH)

#endif

#define TIMER

#define NUM_CHANNELS 3

/* Important data structures */
using pixel_t = uint8_t;

pixel_t *cudaImg;
pixel_t *cudaGrayContent;
int *cudaFinalMatrixCompress;
int *cudaFinalMatrixDecompress;
float *cudaGlobalDCT;

/* Important data structures */
float *globalDCT;
int *finalMatrixCompress;
int *finalMatrixDecompress;

// CUDA functions
// void cudaSetup();
