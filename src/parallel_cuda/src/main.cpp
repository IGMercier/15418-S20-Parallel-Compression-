#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>

#include "../include/config_cuda.hh"
#include "../../../include/stb_image.h"
#include "../../../include/stb_image_write.h"

using namespace std;
using pixel_t = uint8_t;


int main(int argc, char **argv) {
    FILE *fp;
    fp = fopen("./info.txt", "a+"); 
    // TODO: Check if dir exist
    string img_dir = "../../../images/";
    string save_dir = "./../../parallel_omp/compressed_images/";
    string ext = ".jpg";

    string img_name = argv[1] + ext;
    string path = img_dir + img_name;
    cout << img_name << ", ";

#ifdef CUDA
    cout << "CUDA, ";
#endif

    int width, height, bpp;
    pixel_t *const img = stbi_load(path.data(), &width, &height, &bpp, 3);
    cudaSetup(img, width, height);
    auto start = chrono::high_resolution_clock::now();
    compress(width, height);
    auto end = chrono::high_resolution_clock::now();
    cudaFinish(img, width, height);

    std::chrono::duration<double> diff_parallel = end - start;
    cout << "Width: " << width << ", ";
    cout << "Height: " << height << ", ";

#if SERIAL
    string save_img = save_dir + "ser_" + img_name;
    stbi_write_jpg(save_img.data(), width, height, bpp, img, width * bpp);
#else
    string save_img = save_dir + "par_" + img_name;
    stbi_write_jpg(save_img.data(), width, height, bpp, img, width * bpp);
#endif
    stbi_image_free(img);

#if SERIAL
    cout << "Serial -> ";
#else
    cout << "Parallel -> ";
#endif   
    cout << diff_parallel.count() << endl;

    fprintf(fp, "%f ", (float)diff_parallel.count());
    fclose(fp);
    return 0;
}
