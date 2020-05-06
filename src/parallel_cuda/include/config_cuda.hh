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
#endif

/* If CUDA is defined */
#ifdef CUDA

#define BLK_WIDTH 8
#define BLK_HEIGHT 8
#define BLOCKSIZE (BLK_HEIGHT * BLK_WIDTH)

#endif

#define TIMER

#define NUM_CHANNELS 3

/* Important data structures */
// using pixel_t = uint8_t;

extern uint8_t *cudaImg;
// uint8_t *cudaGrayContent;
// int *cudaFinalMatrixCompress;
// int *cudaFinalMatrixDecompress;
// float *cudaGlobalDCT;
// int **cudaQuantArr;

/* Important data structures */
// float *globalDCT;
// int *finalMatrixCompress;
// int *finalMatrixDecompress;
// int **quantArr;

// CUDA functions
void cudaSetup(uint8_t *img, int width, int height);
void cudaFinish(uint8_t *img, int width, int height);
