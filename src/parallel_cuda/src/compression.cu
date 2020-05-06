#include <iostream>
#include <cmath>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/config_cuda.hh"

__constant__ int cudaQuantArr[WINDOW_Y * WINDOW_X];
uint8_t *cudaImg;


__device__
void discreteCosTransformCuda(const int *grayData, float *patchDCT, const int &linearIdx) {
    // Put these in the read only memory.
    float cos1, cos2, temp;
    // Useful constansts
    const float term1 = M_PI / (float)WINDOW_X;
    const float term2 = M_PI / (float)WINDOW_Y;
    const float one_by_root_2 = 1.0 / sqrtf(2);
    const float one_by_root_2N = 1.0 / sqrtf(2 * WINDOW_X);

    int x, y;
    temp = 0.0;
    for (y = 0; y < WINDOW_Y; y++) {
        for (x = 0; x < WINDOW_X; x++) {
            // These cosine values can be pre-stored for a (WINDOW_X, WINDOW_Y)
            // matrix
            // TODO: Verify the order of x and y
            cos1 = cosf(term1 * (y + 0.5) * threadIdx.y);
            cos2 = cosf(term2 * (x + 0.5) * threadIdx.x);
            temp += grayData[y * blockDim.x + x] * cos1 * cos2;
        }
    }

    temp *= one_by_root_2N;
    if (threadIdx.y > 0) {
        temp *= one_by_root_2;
    }

    if (threadIdx.x > 0) {
        temp *= one_by_root_2;
    }

    // TODO: ensure that u + offset < i + pixel and < n
    patchDCT[linearIdx] = (int)temp;
}


__device__ __inline__
void quantizeCuda(const float *patchDCT, int *quantData, const int &linearIdx) {
    // TODO: Use a single array for the purpose of quantizing
    quantData[linearIdx] = (int)roundf((float)patchDCT[linearIdx]
                                       / cudaQuantArr[linearIdx]);
}


__device__ __inline__
void dequantizeCuda(const int *quantData, int *dequantData, const int &linearIdx) {
    // TODO: Use a single array for the purpose of quantizing and dequantizing
    dequantData[linearIdx] = quantData[linearIdx] * cudaQuantArr[linearIdx];
}


__device__
void invDiscreteCosTransformCuda(const int *dequantData, int *patchInverseDCT, int offset, const int &linearIdx) {
    int x, y;
    float cos1, cos2, temp;
    // Useful constants.
    const float term1 = M_PI / (float)WINDOW_X;
    const float term2 = M_PI / (float)WINDOW_Y;
    const float term3 = 2. / (float)WINDOW_X;
    const float term4 = 2. / (float)WINDOW_Y;
    // 1st value
    temp = 1/4. * (float)dequantData[0 * blockDim.x + 0];
    // First column values
    for (y = 1; y < WINDOW_Y; y++) {
        temp += 1/2. * (float)dequantData[y * blockDim.x + 0];
    }
    // First row values
    for (x = 1; x < WINDOW_X; x++) {
        temp += 1/2. * (float)dequantData[0 * blockDim.x + x];
    }

    // if (offset == 202944) {
    //     printf("Partial INV DCT: %f\n", temp);
    // }

    for (y = 1; y < WINDOW_Y; y++) {
        for (x = 1; x < WINDOW_X; x++) {
            cos1 = cosf(term1 * (y + 0.5) * threadIdx.y);
            cos2 = cosf(term2 * (x + 0.5) * threadIdx.x);
            temp += (float)dequantData[y * blockDim.x + x] * cos1 * cos2;
        }
    }

    // if (offset == 202944) {
    //     printf("Partial INV DCT: %d\n", temp);
    // }

    patchInverseDCT[linearIdx] = temp * term3 * term4;
    // if (offset == 202944) {
    //     printf("Partial INV DCT: %d\n", patchInverseDCT[threadIdx.y * blockDim.x + threadIdx.x]);
    // }
}


__device__ __inline__
int getOffset(int width, int i, int j) {
    /**
     *  width: image width
     *  i: pixel row
     *  j: pixel column
     */
    return (i * width + j) * NUM_CHANNELS;
}


__global__
void compressCuda(uint8_t *cudaImg, int width, int height) {
    // ASSUMPTION: 8 x 8 block
    // TODO: Can save memory by using only 2 arrays
    const int num = BLOCKSIZE;
    __shared__ int grayData[num];
    __shared__ float patchDCT[num];
    __shared__ int quantData[num];
    __shared__ int dequantData[num];
    __shared__ int patchInverseDCT[num];

    int add_rows = (PIXEL - (height % PIXEL) != PIXEL ? PIXEL - (height % PIXEL) : 0);
    int add_columns = (PIXEL - (width % PIXEL) != PIXEL ? PIXEL - (width % PIXEL) : 0);

    // padded dimensions to make multiples of patch size
    int _height = height + add_rows;
    int _width = width + add_columns;

    int blockMinX = blockIdx.x * blockDim.x;
    int blockMaxX = blockMinX + blockDim.x;
    int blockMinY = blockIdx.y * blockDim.y;
    int blockMaxY = blockMinY + blockDim.y;

    blockMaxX = min(blockMaxX, _width);
    blockMaxY = min(blockMaxY, _height);

    int pixelX = blockMinX + threadIdx.x;
    int pixelY = blockMinY + threadIdx.y;

    int linearIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int offset = getOffset(width, pixelY, pixelX);

    // Write grayscale data in `grayData` along with zero padding
    if (pixelX < width && pixelY < height) {
        uint8_t *bgrPixel = (uint8_t *) &cudaImg[offset];
        grayData[linearIdx] = (bgrPixel[0] + bgrPixel[1] + bgrPixel[2]) / 3.f;
    } else if (pixelX >= width) {
        grayData[linearIdx] = 0;
    } else if (pixelY >= height) {
        grayData[linearIdx] = 0;
    }

    __syncthreads();
    discreteCosTransformCuda(grayData, patchDCT, linearIdx);
    quantizeCuda(patchDCT, quantData, linearIdx);
    dequantizeCuda(quantData, dequantData, linearIdx);
    __syncthreads();
    invDiscreteCosTransformCuda(dequantData, patchInverseDCT, offset, linearIdx);
    __syncthreads();

    if (pixelX >= width || pixelY >= height) {
        return;
    }

    cudaImg[offset + 0] = patchInverseDCT[linearIdx];
    cudaImg[offset + 1] = patchInverseDCT[linearIdx];
    cudaImg[offset + 2] = patchInverseDCT[linearIdx];
}


void cudaSetup(uint8_t *img, int width, int height) {
    size_t num = NUM_CHANNELS * width * height;
    cudaMalloc(&cudaImg, sizeof(uint8_t) * num);
    cudaMemcpy(cudaImg, img, sizeof(uint8_t) * num, cudaMemcpyHostToDevice);

    int quantArr[WINDOW_Y * WINDOW_X] = {16, 11, 12, 14, 12, 10, 16, 14,
                                         13, 14, 18, 17, 16, 19, 24, 40,
                                         26, 24, 22, 22, 24, 49, 35, 37,
                                         29, 40, 58, 51, 61, 60, 57, 51,
                                         56, 55, 64, 72, 92, 78, 64, 68,
                                         87, 69, 55, 56, 80, 109, 81, 87,
                                         95, 98, 103, 104, 103, 62, 77, 113,
                                         121, 112, 100, 120, 92, 101, 103, 99
                                         };
    cudaMemcpyToSymbol(cudaQuantArr, &quantArr, sizeof(int) * WINDOW_X * WINDOW_Y);

    // TODO: Number of rows and cols should be based on the padded dimensions.
    int rows = (height + BLK_HEIGHT - 1) / BLK_HEIGHT;
    int cols = (width + BLK_WIDTH - 1) / BLK_WIDTH;
    dim3 blockDim(BLK_WIDTH, BLK_HEIGHT);
    dim3 gridDim(cols, rows);

    compressCuda<<<gridDim, blockDim>>>(cudaImg, width, height);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}


void cudaFinish(uint8_t *img, int width, int height) {
    size_t num = NUM_CHANNELS * width * height;
    cudaMemcpy(img, cudaImg, sizeof(uint8_t) * num, cudaMemcpyDeviceToHost);
    cudaFree(cudaImg);
}
