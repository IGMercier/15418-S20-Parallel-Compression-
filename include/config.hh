#pragma once

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define PIXEL 8
#define WINDOW_X 8
#define WINDOW_Y 8

#define SERIAL 0
#if !SERIAL
#define OMP
#endif

// #define TIMER

#define NUM_THREADS 8
#define NUM_CHANNELS 3

/* Important data structures */
std::vector<std::vector<float>> globalDCT;
std::vector<std::vector<int>> finalMatrixCompress;
std::vector<std::vector<int>> finalMatrixDecompress;
