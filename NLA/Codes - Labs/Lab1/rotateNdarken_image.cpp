#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  const char* input_image_path = argv[1];

  // Load the image using stb_image
  int width, height, channels;
  // for greyscale images force to load only one channel
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  
  if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path
              << std::endl;
    return 1;
  }

  std::cout << "Image loaded: " << width << "x" << height << " with "
            << channels << " channels." << std::endl;

  // Prepare Eigen matrices 
  MatrixXd dark(height, width), light(height, width), rotate(width, height);

  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
      dark(i, j) = std::max(static_cast<double>(image_data[index]) - 50.,0.0) / 255.0;
      light(i, j) = std::min(static_cast<double>(image_data[index]) + 50.,255.) / 255.0;
      rotate(width-j-1, i) = static_cast<double>(image_data[index]) / 255.0;
    }
  }
  // Free memory!!!
  stbi_image_free(image_data);

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> dark_image(height, width);
  // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
  dark_image = dark.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  // Save the image using stb_image_write
  const std::string output_image_path1 = "dark_image.png";
  if (stbi_write_png(output_image_path1.c_str(), width, height, 1,
                     dark_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;

    return 1;
  }

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> rotate_image(width, height);
  // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
  rotate_image = rotate.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  // Save the image using stb_image_write
  const std::string output_image_path2 = "rotate_image.png";
  if (stbi_write_png(output_image_path2.c_str(), height, width, 1,
                     rotate_image.data(), height) == 0) {
    std::cerr << "Error: Could not save output image" << std::endl;
    
    return 1;
  }

  std::cout << "Images saved to " << output_image_path1 << " and " << output_image_path2 << std::endl;

  return 0;
}
