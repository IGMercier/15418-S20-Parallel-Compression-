#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"


int main() {
    int width, height, bpp;
    uint8_t *img_par = stbi_load("sample_par.jpg", &width, &height, &bpp, 3);
    int _width, _height, _bpp;
    uint8_t *img_ser = stbi_load("sample_ser.jpg", &_width, &_height, &_bpp, 3);
    if (width != _width || height != _height || bpp != _bpp) {
        std::cout << "INCORRECT dimensions" << std::endl;
        return 0;
    }
    long error = 0;
    for (int i = 0; i < width * height * bpp; i++)
    {
        error += abs(img_par[i] != img_ser[i]);
    }

    if (error == 0) {
        std::cout << "CORRECT" << std::endl;
    } else {
        std::cout << "INCORRECT: " << error << std::endl;
    }

    return 0;
}
