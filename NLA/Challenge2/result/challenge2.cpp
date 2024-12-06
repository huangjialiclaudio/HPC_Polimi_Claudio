#include "utils.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include <unsupported/Eigen/SparseExtra>

#include <cstdlib>
#include <string>

int main(int argc, char *argv[]) {

    // Initialize the logger
    plog::init(plog::debug, "../ch2_result/log.txt");

    // Load the iamge as matrix A with size m times n
    int width, height;
    auto *image_input_path = "/Users/raopend/Workspace/NLA_challenge/photos/256px-Albert_Einstein_Head.jpg";
    Eigen::MatrixXd image_matrix;
    if (int channels; loadImage(image_input_path, image_matrix, width, height, channels)) {
        PLOG_INFO << "Image loaded successfully.";
    } else {
        PLOG_ERROR << "Failed to load the image.";
        return EXIT_FAILURE;
    }

    // Report the size of the matrix
    PLOG_INFO << "The size of the original image matrix is: " + std::to_string(height) + " x " + std::to_string(width);

    Eigen::MatrixXd A = image_matrix;

    // compute Gram matrix
    const Eigen::MatrixXd gram_matrix = A.transpose() * A;

    // Report the euclidean norm of the Gram matrix
    const double euclidean_norm = gram_matrix.norm();
    PLOG_INFO << "The Euclidean norm of the Gram matrix is: " + std::to_string(euclidean_norm);

    // Solve the eigenvalues of the Gram matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(gram_matrix);
    const Eigen::VectorXd eigenvalues = es.eigenvalues().real();
    const Eigen::MatrixXd eigenvectors = es.eigenvectors().real();

    // Report the two largest eigenvalues
    PLOG_INFO << "The two largest eigenvalues are: " + std::to_string(eigenvalues(eigenvalues.size() - 1)) + " and " +
                         std::to_string(eigenvalues(eigenvalues.size() - 2));

    // Export gram matrix to a .mtx file
    saveMarket(gram_matrix, "../ch2_result/gram_matrix.mtx");

    // perform a singular value decomposition of thematrix A.
    Eigen::JacobiSVD<Eigen::MatrixXd> jcb_svd;
    jcb_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    PLOG_INFO << "The largest singular values of the matrix A are: " + std::to_string(jcb_svd.singularValues()(0)) +
                         " and " + std::to_string(jcb_svd.singularValues()(1)) + " using Jacobi SVD.";
    Eigen::BDCSVD<Eigen::MatrixXd> bdcs_svd;
    bdcs_svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    PLOG_INFO << "The largest singular values of the matrix A are: " + std::to_string(bdcs_svd.singularValues()(0)) +
                         " and " + std::to_string(bdcs_svd.singularValues()(1)) + " using BDC SVD.";
    // Get the diagonal matrix \Sigma
    const Eigen::MatrixXd Sigma = bdcs_svd.singularValues().asDiagonal();

    // Report the norm of \Sigma
    PLOG_INFO << "The norm of the diagonal matrix Sigma is: " + std::to_string(Sigma.norm());

    // use truncated SVD k = 40
    constexpr int k1{40};
    const Eigen::MatrixXd C1 = bdcs_svd.matrixU().leftCols(k1);
    const Eigen::MatrixXd D1 = Sigma.topLeftCorner(k1, k1) * bdcs_svd.matrixV().leftCols(k1).transpose();
    // Report the number of nonzero entries in the matrices C1 and D1
    PLOG_INFO << "The number of nonzero entries in the matrix C1 is: " + std::to_string(C1.nonZeros());
    PLOG_INFO << "The number of nonzero entries in the matrix D1 is: " + std::to_string(D1.nonZeros());

    // use truncated SVD k = 80
    constexpr int k2{80};
    const Eigen::MatrixXd C2 = bdcs_svd.matrixU().leftCols(k2);
    const Eigen::MatrixXd D2 = Sigma.topLeftCorner(k2, k2) * bdcs_svd.matrixV().leftCols(k2).transpose();
    // Report the number of nonzero entries in the matrices C
    PLOG_INFO << "The number of nonzero entries in the matrix C2 is: " + std::to_string(C2.nonZeros());
    PLOG_INFO << "The number of nonzero entries in the matrix D2 is: " + std::to_string(D2.nonZeros());

    // Compute the compressed images as thematrix product
    const Eigen::MatrixXd A_tile1 = C1 * D1;
    const Eigen::MatrixXd A_tile2 = C2 * D2;

    // Export the images to .png files
    saveImage("../ch2_result/compressed_image_k40.png", A_tile1, height, width);
    saveImage("../ch2_result/compressed_image_k80.png", A_tile2, height, width);
    saveImage("../ch2_result/original_image.png", A, height, width);

    // Using Eigen create a black and white checkerboard image of size 200x200 pixels, ranging from 0 to 255
    constexpr int checkerboard_size{200};
    Eigen::MatrixXd checkerboard_image(checkerboard_size, checkerboard_size);
    for (int i = 0; i < checkerboard_size; ++i) {
        for (int j = 0; j < checkerboard_size; ++j) {
            // checkerboard_image(i, j) = (i + j) % 2 == 0 ? 0 : 255;
            // 8 timms 8 checkerboard
            checkerboard_image(i, j) = ((i / 25) + (j / 25)) % 2 == 0 ? 0 : 255;
        }
    }
    // Export the checkerboard image to a .png file
    saveImage("../ch2_result/checkerboard_image.png", checkerboard_image, checkerboard_size, checkerboard_size);
    // Report the Euclidean norm
    PLOG_INFO << "The Euclidean norm of the checkerboard image is: " + std::to_string(checkerboard_image.norm());

    // Introduce a noise into the checkerboard image by adding random fluctuations of color ranging between[− 50, 50] to
    // each pixel.
    Eigen::MatrixXd noisy_checkerboard_image =
            checkerboard_image + Eigen::MatrixXd::Random(checkerboard_size, checkerboard_size) * 50;
    // Export the noisy checkerboard image to a .png file
    saveImage("../ch2_result/noisy_checkerboard_image.png", noisy_checkerboard_image, checkerboard_size,
              checkerboard_size);

    // SVD of the nosiy checkerboard image
    Eigen::JacobiSVD<Eigen::MatrixXd> jcb_svd_noisy;
    jcb_svd_noisy.compute(noisy_checkerboard_image, Eigen::ComputeThinU | Eigen::ComputeThinV);
    PLOG_INFO << "The largest singular values of the noisy checkerboard image are: " +
                         std::to_string(jcb_svd_noisy.singularValues()(0)) + " and " +
                         std::to_string(jcb_svd_noisy.singularValues()(1)) + " using Jacobi SVD.";
    const Eigen::MatrixXd Sigma_noisy = jcb_svd_noisy.singularValues().asDiagonal();
    // Perform a truncated SVD with k =5 and k = 10
    constexpr int k3{5};
    const Eigen::MatrixXd C3 = jcb_svd_noisy.matrixU().leftCols(k3);
    const Eigen::MatrixXd D3 = Sigma_noisy.topLeftCorner(k3, k3) * jcb_svd_noisy.matrixV().leftCols(k3).transpose();
    // Report the size of the matrices C and D.
    PLOG_INFO << "The size of the matrix C3 is: " + std::to_string(C3.rows()) + " x " + std::to_string(C3.cols());
    PLOG_INFO << "The size of the matrix D3 is: " + std::to_string(D3.rows()) + " x " + std::to_string(D3.cols());

    constexpr int k4{10};
    const Eigen::MatrixXd C4 = jcb_svd_noisy.matrixU().leftCols(k4);
    const Eigen::MatrixXd D4 = Sigma_noisy.topLeftCorner(k4, k4) * jcb_svd_noisy.matrixV().leftCols(k4).transpose();
    // Report the size of the matrices C and D.
    PLOG_INFO << "The size of the matrix C4 is: " + std::to_string(C4.rows()) + " x " + std::to_string(C4.cols());
    PLOG_INFO << "The size of the matrix D4 is: " + std::to_string(D4.rows()) + " x " + std::to_string(D4.cols());

    // Compute the compressed images as the matrix product
    const Eigen::MatrixXd A_tile3 = C3 * D3;
    const Eigen::MatrixXd A_tile4 = C4 * D4;

    // Export the images to .png files
    saveImage("../ch2_result/compressed_noisy_checkerboard_image_k5.png", A_tile3, checkerboard_size,
              checkerboard_size);
    saveImage("../ch2_result/compressed_noisy_checkerboard_image_k10.png", A_tile4, checkerboard_size,
              checkerboard_size);

    // comment
    /*
     * The compressed images we obtained are very similar to the original image.
     * Using the truncated SVD method, we discarded the smaller singular values, image compression becomes lossy, but
     * good image compression is virtually undetectable by the human visual system. We also noticed that SVD can also be
     * used to denoise images, because the rank of checkerboard is 2,it is possible to construct a low-rank
     * approximation matrix with two pieces when the noise is really small.
     */
    return EXIT_SUCCESS;
}
