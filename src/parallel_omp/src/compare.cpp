#include <iostream>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../../include/stb_image.h"
#include "../../../include/stb_image_write.h"


bool same(std::string &save_dir, std::string &img1, std::string &img2) {
    // Load imagges
    int width, height, bpp;
    uint8_t *img_par = stbi_load((save_dir + img1).data(), &width, &height, &bpp, 3);
    int _width, _height, _bpp;
    uint8_t *img_ser = stbi_load((save_dir + img2).data(), &_width, &_height, &_bpp, 3);

    // Check dimensions
    if (width != _width || height != _height || bpp != _bpp) {
        std::cout << "INCORRECT dimensions: " << img1 << " and " << img2;
        std::cout << std::endl;
        return false;
    }

    // Check pixel values
    long error = 0;
    int max_diff = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uint8_t *bgrPixelPar = img_par + (i * width + j) * 3;
            uint8_t *bgrPixelSer = img_ser + (i * width + j) * 3;
            if (bgrPixelPar[0] != bgrPixelPar[1] || bgrPixelPar[0] != bgrPixelPar[2]
                || bgrPixelSer[0] != bgrPixelSer[1] || bgrPixelSer[0] != bgrPixelSer[2]) {
                std::cout << "Pixel values across channels are not same" << std::endl;
            }

            max_diff = std::max(max_diff, abs(bgrPixelPar[0] - bgrPixelSer[0]));
            error += !!abs(bgrPixelPar[0] - bgrPixelSer[0]);
            // Print information about the mismatching pixels
            if (bgrPixelPar[0] != bgrPixelSer[0]) {
                // std::cout << "H: " << i << " W: " << j << std::endl;
                // printf("Parallel: %d\n", bgrPixelPar[0]);
                // printf("Serial: %d\n", bgrPixelSer[0]);
                // break;
            }
        }
    }

    if (error == 0) {
        std::cout << "CORRECT" << std::endl;
    } else {
        std::cout << "INCORRECT: " << img1 << " and " << img2;
        std::cout << " | Num different values: " << error << " | Max diff: " << max_diff;
        std::cout << std::endl;
    }

    return error == 0;
}


int main() {
    std::cout << "*********" << " COMPARISON " << "*********" << std::endl;
    std::vector<std::string> par_files, ser_files;
    std::string file = "", refined_file = "";
    std::string save_dir = "./compressed_images/";

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (save_dir.data())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            file = ent->d_name;
            refined_file = "";
            for (auto &c: file) {
                if (c == ' ')
                    continue;
                refined_file += c;
            }
            if (refined_file.substr(0, 3) == "ser") {
                ser_files.push_back(refined_file);
            } else if (refined_file.substr(0, 3) == "par") {
                par_files.push_back(refined_file);
            }
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }

    sort(ser_files.begin(), ser_files.end());
    sort(par_files.begin(), par_files.end());

    if (ser_files.size() != par_files.size()) {
        std::cout << "Unequal amount of serial and parallel files" << std::endl;
        return 0;
    }

    for (int i = 0; i < ser_files.size(); i++) {
        auto img1 = par_files[i];
        auto img2 = ser_files[i];
        if (img1.length() != img2.length()) {
            std::cout << "Different images" << std::endl;
        }

        std::string p = "", s = "";
        for (int i = 0; i < img1.length(); i++) {
            if (isdigit(img1[i])) {
                p += img1[i];
            }
            if (isdigit(img2[i])) {
                s += img2[i];
            }
        }

        if (s != p) {
            std::cout << "Different images" << std::endl;
        }

        if (!same(save_dir, par_files[i], ser_files[i])) {
            // break;
        }
    }
    
    return 0;
}
