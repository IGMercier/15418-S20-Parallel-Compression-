#include <cmath>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>

#include "../include/config_cuda.hh"

__constant__ int cudaQuantArr[WINDOW_Y * WINDOW_X];
__constant__ float cudaCosArr1[WINDOW_Y * WINDOW_X];
__constant__ float cudaCosArr2[WINDOW_Y * WINDOW_X];
__constant__ float cudaOne_by_root_2;
__constant__ float cudaOne_by_root_2N;
__constant__ float cudaTerm3;
__constant__ float cudaTerm4;
uint8_t *cudaImg;


__device__
void discreteCosTransformCuda(const int *grayData, float *patchDCT, int offset, const int &linearIdx) {
    float cos1, cos2, temp;
    int tx = threadIdx.x % WINDOW_X;
    int ty = threadIdx.y % WINDOW_Y;
    int offsetX = (int)(threadIdx.x / WINDOW_X) * WINDOW_X;
    int offsetY = (int)(threadIdx.y / WINDOW_Y) * WINDOW_Y;
    int x, y;
    temp = 0.0;
    // Loop over all pixels in patch
    for (y = 0; y < WINDOW_Y; y++) {
        for (x = 0; x < WINDOW_X; x++) {
            cos1 = cudaCosArr1[y * WINDOW_X + ty];
            cos2 = cudaCosArr2[x * WINDOW_X + tx];
            temp += grayData[(y + offsetY) * blockDim.x + (x + offsetX)] * cos1 * cos2;
        }
    }

    temp *= cudaOne_by_root_2N;
    if (ty > 0) {
        temp *= cudaOne_by_root_2;
    }

    if (tx > 0) {
        temp *= cudaOne_by_root_2;
    }

    patchDCT[linearIdx] = (int)temp;
}


__device__ __inline__
void quantizeCuda(const float *patchDCT, int *quantData, const int &linearIdx) {
    int tx = threadIdx.x % WINDOW_X;
    int ty = threadIdx.y % WINDOW_Y;
    quantData[linearIdx] = (int)roundf((float)patchDCT[linearIdx]
                                       / cudaQuantArr[ty * WINDOW_Y + tx]);
}


__device__ __inline__
void dequantizeCuda(const int *quantData, int *dequantData, const int &linearIdx) {
    int tx = threadIdx.x % WINDOW_X;
    int ty = threadIdx.y % WINDOW_Y;
    dequantData[linearIdx] = quantData[linearIdx] * cudaQuantArr[ty * WINDOW_Y + tx];
}


__device__
void invDiscreteCosTransformCuda(const int *dequantData, int *patchInverseDCT, int offset, const int &linearIdx) {
    int x, y;
    float cos1, cos2, temp;
    int tx = threadIdx.x % WINDOW_X;
    int ty = threadIdx.y % WINDOW_Y;
    int offsetX = (int)(threadIdx.x / WINDOW_X) * WINDOW_X;
    int offsetY = (int)(threadIdx.y / WINDOW_Y) * WINDOW_Y;

    // 1st value
    temp = 1/4. * (float)dequantData[(0 + offsetY) * blockDim.x + (0 + offsetX)];
    // First column values
    for (y = 1; y < WINDOW_Y; y++) {
        temp += 1/2. * (float)dequantData[(y + offsetY) * blockDim.x + (0 + offsetX)];
    }
    // First row values
    for (x = 1; x < WINDOW_X; x++) {
        temp += 1/2. * (float)dequantData[(0 + offsetY) * blockDim.x + (x + offsetX)];
    }
    // Loop over all pixels in patch
    for (y = 1; y < WINDOW_Y; y++) {
        for (x = 1; x < WINDOW_X; x++) {
            cos1 = cudaCosArr1[y * WINDOW_X + ty];
            cos2 = cudaCosArr2[x * WINDOW_X + tx];
            temp += (float)dequantData[(y + offsetY) * blockDim.x + (x + offsetX)] * cos1 * cos2;
        }
    }

    patchInverseDCT[linearIdx] = temp * cudaTerm3 * cudaTerm4;
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
    __shared__ int grayData[BLOCKSIZE];
    __shared__ float patchDCT[BLOCKSIZE];
    __shared__ int quantData[BLOCKSIZE];
    __shared__ int dequantData[BLOCKSIZE];
    __shared__ int patchInverseDCT[BLOCKSIZE];

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

    discreteCosTransformCuda(grayData, patchDCT, offset, linearIdx);
    quantizeCuda(patchDCT, quantData, linearIdx);
    dequantizeCuda(quantData, dequantData, linearIdx);

    __syncthreads();

    invDiscreteCosTransformCuda(dequantData, patchInverseDCT, offset, linearIdx);

    __syncthreads();

    if (pixelX >= width || pixelY >= height) {
        return;
    }

    uint8_t pixelValue = patchInverseDCT[linearIdx];
    cudaImg[offset + 0] = pixelValue;
    cudaImg[offset + 1] = pixelValue;
    cudaImg[offset + 2] = pixelValue;
}


void cudaSetup(uint8_t *img, int width, int height) {
    /* Allocate data structure for storing the image on device global memory */
    size_t num = NUM_CHANNELS * width * height;
    cudaMalloc(&cudaImg, sizeof(uint8_t) * num);
    cudaMemcpy(cudaImg, img, sizeof(uint8_t) * num, cudaMemcpyHostToDevice);

    /* Store constants in the device global read-only memory */
    int quantArr[WINDOW_Y * WINDOW_X] = {16, 11, 12, 14, 12, 10, 16, 14,
                                         13, 14, 18, 17, 16, 19, 24, 40,
                                         26, 24, 22, 22, 24, 49, 35, 37,
                                         29, 40, 58, 51, 61, 60, 57, 51,
                                         56, 55, 64, 72, 92, 78, 64, 68,
                                         87, 69, 55, 56, 80, 109, 81, 87,
                                         95, 98, 103, 104, 103, 62, 77, 113,
                                         121, 112, 100, 120, 92, 101, 103, 99};
    float cosArr1[WINDOW_Y * WINDOW_X];
    float cosArr2[WINDOW_Y * WINDOW_X];
    for (int i = 0; i < WINDOW_Y; i++) {
        for (int j = 0; j < WINDOW_X; j++) {
            cosArr1[i * WINDOW_X + j] = cos(term1 * (i + 0.5) * j);
            cosArr2[i * WINDOW_X + j] = cos(term2 * (i + 0.5) * j);
        }
    }

    cudaMemcpyToSymbol(cudaQuantArr, &quantArr, sizeof(int) * WINDOW_X * WINDOW_Y);
    cudaMemcpyToSymbol(cudaCosArr1, &cosArr1, sizeof(float) * WINDOW_X * WINDOW_Y);
    cudaMemcpyToSymbol(cudaCosArr2, &cosArr2, sizeof(float) * WINDOW_X * WINDOW_Y);
    cudaMemcpyToSymbol(cudaOne_by_root_2, &one_by_root_2, sizeof(float));
    cudaMemcpyToSymbol(cudaOne_by_root_2N, &one_by_root_2N, sizeof(float));
    cudaMemcpyToSymbol(cudaTerm3, &term3, sizeof(float));
    cudaMemcpyToSymbol(cudaTerm4, &term4, sizeof(float));
}


void compress(int width, int height) {
    // TODO: Number of rows and cols should be based on the padded dimensions.
    // Or not?
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
