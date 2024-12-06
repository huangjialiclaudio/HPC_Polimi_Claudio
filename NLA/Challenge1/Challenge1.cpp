#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;



int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    // step1: Load the image using stb_image
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  // Force load as RGB

    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
    
    MatrixXd originalImage(height,width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = i * width + j;
            originalImage(i, j) = static_cast<double>(image_data[index]);
        }
    }

    // Free memory!!!

    //stbi_image_free(image_data);

    // step2: Introduce a noise signal
    Eigen::MatrixXd noiseImage(height,width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double x = rand() % 100 - 50;
            if(originalImage(i,j) + x < 0.0 )
                noiseImage(i,j) = 0.0;
            else if (originalImage(i,j) + x > 255.0)
                noiseImage(i,j) = 255.0;
            else
                noiseImage(i,j) = originalImage(i,j) + x;


        }
    }
    Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noiseImage_result(height, width);

    noiseImage_result = noiseImage.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val);
    });
    // Save the grayscale image using stb_image_write
    const std::string output_image_path = "noiseImage.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                        noiseImage_result.data(), width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;

        return 1;
    }

    std::cout << "noiseImage_result saved to " << output_image_path << std::endl;


    //  step3: Reshape the original and noisy images
    VectorXd v(height*width);
    for(int i = 0;i < height; ++i){
        for(int j = 0;j < width; ++j){
            v(i*height+j) = originalImage(i,j);
        }
    }



    return 0;
}

