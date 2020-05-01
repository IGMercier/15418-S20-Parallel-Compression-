#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#define pixel 8
#define WINDOW 8

using namespace std;

vector<vector<int>> quantArr = {{16, 11, 12, 14, 12, 10, 16, 14},
                                {13, 14, 18, 17, 16, 19, 24, 40},
                                {26, 24, 22, 22, 24, 49, 35, 37},
                                {29, 40, 58, 51, 61, 60, 57, 51},
                                {56, 55, 64, 72, 92, 78, 64, 68},
                                {87, 69, 55, 56, 80, 109, 81, 87},
                                {95, 98, 103, 104, 103, 62, 77, 113},
                                {121, 112, 100, 120, 92, 101, 103, 99}
                                };

// float globalDCT[3005][3005];
// vector<vector<float>> globalDCT;
float **globalDCT{};

struct INF_COMPRESS {
  vector<int> v;
} finalMatrixCompress[3000][3000];
int finalMatrixDecompress[3000][3000];

// vector<vector<vector<int>>> finalMatrixCompress(200, vector<vector<int>>(200, vector<int>(64)));
// vector<vector<int>> finalMatrixDecompress(1000, vector<int>(1000));

/** Quantize a block by dividing its pixel value with the respective value
 * in the quantization matrix
 */
void quantizeBlock(int R, int C) {
    int i, j;
    vector<int> vRLE(WINDOW * WINDOW);

    for (i = 0; i < WINDOW; i++) {
        for (j = 0; j < WINDOW; j++) {
            int temp = globalDCT[R * WINDOW + i][C * WINDOW + j];
            temp = (int)round((float)temp / quantArr[i][j]);
            vRLE[i * WINDOW + j] = temp;
        }
    }

    finalMatrixCompress[R + 1][C + 1].v = vRLE;
    // finalMatrixCompress[R + 1][C + 1] = vRLE;
    return;
}


void quantize(int height, int width) {
    int i, j;

#if !SERIAL
#ifdef OMP
    #pragma omp parallel for schedule(runtime)
#endif
#endif
    for (i = 0; i < height / WINDOW; i++) {
        for (j = 0; j < width / WINDOW; j++) {
            quantizeBlock(i, j);
        }
    }

    return;
}


