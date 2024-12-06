#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include "utils.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include "lis.h"

#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

using namespace Eigen;


// Function to count only the truly non-zero elements in a sparse matrix
template<typename T>
int countNonZeroElements(const SparseMatrix<T> &matrix) {
    int count = 0;
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (typename SparseMatrix<T>::InnerIterator it(matrix, k); it; ++it) {
            if (it.value() != 0.0) { // Only count true non-zero elements
                ++count;
            }
        }
    }
    return count;
}

// Function to create a sparse matrix representing the A_avg 2 smoothing kernel
SparseMatrix<double> createAAvg2Matrix(const int height, const int width) {
    const long size = height * width;
    SparseMatrix<double> S(size, size);
    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(size * 9); // Each pixel has up to 9 neighbors

    // Iterate over every pixel in the image
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            long currentIndex = i * width + j;

            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ni = i + di;
                    if (const int nj = j + dj; ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        int neighborIndex = ni * width + nj;
                        tripletList.emplace_back(currentIndex, neighborIndex, 1.0 / 9.0);
                    }
                }
            }
        }
    }

    S.setFromTriplets(tripletList.begin(), tripletList.end());

    return S;
}

// Function to create a sparse matrix representing the A_avg 2 smoothing kernel using matrix shifts
SparseMatrix<double> createAAvg2MatrixOptimized(const int height, const int width) {
    const int size = height * width;
    SparseMatrix<double> S(size, size);

    // Identity matrix to represent the base image (center pixel of the kernel)
    SparseMatrix<double> I(size, size);
    I.setIdentity();

    // Define shifts in 8 different directions (left, right, up, down, diagonals)
    SparseMatrix<double> shiftUp(size, size), shiftDown(size, size);
    SparseMatrix<double> shiftLeft(size, size), shiftRight(size, size);
    SparseMatrix<double> shiftUpLeft(size, size), shiftUpRight(size, size);
    SparseMatrix<double> shiftDownLeft(size, size), shiftDownRight(size, size);

    // Shift by one row (up and down)
    shiftUp.reserve(size);
    shiftDown.reserve(size);
    for (int i = width; i < size; ++i) {
        shiftUp.insert(i, i - width) = 1.0;
        shiftDown.insert(i - width, i) = 1.0;
    }

    // Shift by one column (left and right)
    shiftLeft.reserve(size);
    shiftRight.reserve(size);
    for (int i = 1; i < size; ++i) {
        if (i % width != 0) { // avoid crossing row boundaries
            shiftLeft.insert(i, i - 1) = 1.0;
            shiftRight.insert(i - 1, i) = 1.0;
        }
    }

    // Diagonal shifts (up-left, up-right, down-left, down-right)
    shiftUpLeft.reserve(size);
    shiftUpRight.reserve(size);
    shiftDownLeft.reserve(size);
    shiftDownRight.reserve(size);
    for (int i = width; i < size; ++i) {
        if (i % width != 0) { // avoid crossing row boundaries
            shiftUpLeft.insert(i - width - 1, i) = 1.0;
            shiftDownRight.insert(i, i - width - 1) = 1.0;
        }
        if (i % width != width - 1) { // avoid crossing row boundaries
            shiftUpRight.insert(i - width + 1, i) = 1.0;
            shiftDownLeft.insert(i, i - width + 1) = 1.0;
        }
    }

    // Combine all shifted matrices with equal weights (1/9 for average smoothing)
    S = (I + shiftUp + shiftDown + shiftLeft + shiftRight + shiftUpLeft + shiftUpRight + shiftDownLeft +
         shiftDownRight) /
        (1.0 / 9.0);

    return S;
}

// Function to create a sparse matrix representing the H_sh2 sharpening kernel
SparseMatrix<double> createHsh2Matrix(const int height, const int width) {
    const long size = height * width; // Total number of pixels in the image
    SparseMatrix<double> S(size, size);
    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(size * 9); // Each pixel has up to 9 neighbors

    // Define the sharpening filter H_sh2
    constexpr int filter[3][3] = {{0, -3, 0}, {-1, 9, -3}, {0, -1, 0}};

    // Iterate over every pixel in the image
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            long currentIndex = i * width + j;

            // Add weights of neighboring pixels (3x3 window)
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ni = i + di; // Neighbor row index
                    if (const int nj = j + dj; ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        long neighborIndex = ni * width + nj;
                        double weight = filter[di + 1][dj + 1]; // Adjust index to filter space
                        tripletList.emplace_back(currentIndex, neighborIndex, weight);
                    }
                }
            }
        }
    }

    // Build the sparse matrix from the triplet list
    S.setFromTriplets(tripletList.begin(), tripletList.end());

    return S;
}

// Function to create a sparse matrix representing the H_sh2 sharpening kernel using matrix shifts
SparseMatrix<double> createHsh2MatrixOptimized(const int height, const int width) {
    const int size = height * width; // Total number of pixels
    SparseMatrix<double> S(size, size);

    // Identity matrix to represent the base image (center element in the filter)
    SparseMatrix<double> I(size, size);
    I.setIdentity();

    // Define shifts in 8 different directions (up, down, left, right, diagonals)
    SparseMatrix<double> shiftUp(size, size), shiftDown(size, size);
    SparseMatrix<double> shiftLeft(size, size), shiftRight(size, size);

    // Shift by one row (up and down)
    shiftUp.reserve(size);
    shiftDown.reserve(size);
    for (int i = width; i < size; ++i) {
        shiftUp.insert(i, i - width) = 1.0;
        shiftDown.insert(i - width, i) = 1.0;
    }

    // Shift by one column (left and right)
    shiftLeft.reserve(size);
    shiftRight.reserve(size);
    for (int i = 1; i < size; ++i) {
        if (i % width != 0) {
            shiftLeft.insert(i, i - 1) = 1.0;
            shiftRight.insert(i - 1, i) = 1.0;
        }
    }


    // Apply weights from the sharpening filter H_sh2
    S = I * 9.0 // center pixel weight
        + shiftUp * (-1.0) + shiftDown * (-1.0) + shiftLeft * (-3.0) + shiftRight * (-3.0);

    return S;
}

// Function to create a sparse matrix representing the Laplacian filter (0, -1, 0; -1, 4, -1; 0, -1, 0)
SparseMatrix<double> createLaplacianMatrixOptimized(const int height, const int width) {
    const int size = height * width; // Total number of pixels in the image
    SparseMatrix<double> S(size, size);

    // Identity matrix to represent the base image (center element in the filter)
    SparseMatrix<double> I(size, size);
    I.setIdentity();

    // Define shifts in 4 directions (up, down, left, right)
    SparseMatrix<double> shiftUp(size, size), shiftDown(size, size);
    SparseMatrix<double> shiftLeft(size, size), shiftRight(size, size);

    // Shift by one row (up and down)
    shiftUp.reserve(size);
    shiftDown.reserve(size);
    for (int i = width; i < size; ++i) {
        shiftUp.insert(i, i - width) = 1.0;
        shiftDown.insert(i - width, i) = 1.0;
    }

    // Shift by one column (left and right)
    shiftLeft.reserve(size);
    shiftRight.reserve(size);
    for (int i = 1; i < size; ++i) {
        if (i % width != 0) {
            shiftLeft.insert(i, i - 1) = 1.0;
            shiftRight.insert(i - 1, i) = 1.0;
        }
    }

    // Apply weights from the Laplacian filter
    S = I * 4.0 // center pixel weight (4)
        + shiftUp * -1.0 // up
        + shiftDown * -1.0 // down
        + shiftLeft * -1.0 // left
        + shiftRight * -1.0 // right
            ;

    return S;
}


int main(int argc, char *argv[]) {
    // Initialize the logger
    plog::init(plog::debug, "../ch1_result/log.txt");
    /**
     * Load the image as an Eigen matrix with size m × n.
     * Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 (black) and
     * 255 (white). Report the size of the matrix.
     */

    // Load the image as an Eigen matrix with size m × n.
    int width, height, channels;
    auto *input_image_path = "/Users/raopend/Workspace/NLA_challenge/photos/256px-Albert_Einstein_Head.jpg";
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 3); // Force load as grayscale

    if (!image_data) {
        LOG_ERROR << "Could not load image";
        return 1;
    }
    // Prepare Eigen matrices for each RGB channel
    MatrixXd red(height, width), green(height, width), blue(height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            const int index = (i * width + j) * 3; // 3 channels (RGB)
            red(i, j) = static_cast<double>(image_data[index]) / 255.0;
            green(i, j) = static_cast<double>(image_data[index + 1]) / 255.0;
            blue(i, j) = static_cast<double>(image_data[index + 2]) / 255.0;
        }
    }
    // Free memory!!!
    stbi_image_free(image_data);

    // Create a grayscale matrix
    const MatrixXd gray = convertToGrayscale(red, green, blue);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> grayscale_image_matrix(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    grayscale_image_matrix =
            gray.unaryExpr([](const double val) -> unsigned char { return static_cast<unsigned char>(val * 255.0); });

    // Report the size of the matrix
    PLOG_INFO << "The size of the original image matrix is: " + std::to_string(height) + " x " + std::to_string(width);


    /**
     *Introduce a noise signal into the loaded image by adding random fluctuations of color
     *ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.
     */

    // generate the random matrix, color ranging between -50 and 50
    MatrixXd noise_matrix = MatrixXd::Random(grayscale_image_matrix.rows(), grayscale_image_matrix.cols());
    noise_matrix = 50 * noise_matrix;
    // add the noise to the image matrix
    MatrixXd noisy_image_matrix = grayscale_image_matrix.cast<double>() + noise_matrix;
    // Save the grayscale image using stb_image_write
    const std::string output_image_path = "../ch1_result/output_noisy.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1, convertToUnsignedChar(noisy_image_matrix).data(),
                       width) == 0) {
        PLOG_ERROR << "Could not save noisy image";
        return 1;
    }
    PLOG_INFO << "Noisy image saved to: " + output_image_path;


    /**
     * Reshape the original image matrix and nosiy image matrix into vectors \vec{v} and \vec{w} respectively.
     * Verify that each vector has mn components. Report here the Euclidean norm of \vec{v}.
     */

    // Reshape the original image matrix and noisy  image matrix into vectors
    VectorXd v = grayscale_image_matrix.cast<double>().reshaped<RowMajor>().transpose();
    // Verify that each vector has mn components
    assert(v.size() == grayscale_image_matrix.size());
    // Verify that each vector has mn components
    VectorXd w = noisy_image_matrix.reshaped<RowMajor>().transpose();
    assert(w.size() == noisy_image_matrix.size());


    // Report here the Euclidean norm of \vec{v}
    PLOG_INFO << "The Euclidean norm of v is: " + std::to_string(v.norm());


    /**
     * Write the convolution operation corresponding to the smoothing kernel Hav2
     * as a matrix vector multiplication between a matrix A1 having size mn × mn and the image vector.
     * Report the number of non-zero entries in A1.
     */

    // Define the smoothing kernel Hav2
    // Define the matrix A1
    auto A1 = createAAvg2Matrix(height, width);
    PLOG_INFO << "The number of non-zero entries in A1 is: " + std::to_string(countNonZeroElements(A1));
    /**
     * Apply the previous smoothing filter to the noisy image by performing the matrix vector multiplication A1w.
     * Export and upload the resulting image.
     */

    // Apply the smoothing filter to the noisy image
    auto smoothed_image = A1 * w;
    // Reshape the smoothed image vector to a matrix
    auto smoothed_image_matrix = smoothed_image.reshaped<RowMajor>(height, width);
    // Save the smoothed image using stb_image_write
    const std::string smoothed_image_path = "../ch1_result/output_smoothed.png";
    if (stbi_write_png(smoothed_image_path.c_str(), width, height, 1,
                       convertToUnsignedChar(smoothed_image_matrix).data(), width) == 0) {
        PLOG_ERROR << "Could not save smoothed image";
        return 1;
    }
    PLOG_INFO << "Smoothed image saved to: " + smoothed_image_path;

    /**
     * Write the convolution operation corresponding to the sharpening kernel Hsh2
     * as a matrix vector multiplication by a matrix A2 having size mn × mn. Report the number of non-zero
     * entries in A2. Is A2 symmetric?
     */
    auto A2 = createHsh2Matrix(height, width);
    PLOG_INFO << "The number of non-zero entries in A2 is: " + std::to_string(countNonZeroElements(A2));
    // Report if matrix A2 is symmetric
    PLOG_INFO << "Matrix A2 is symmetric: " + std::to_string(A2.isApprox(A2.transpose()));


    /**
     * Apply the previous sharpening filter to the original image by performing the matrix vector multiplication A2v.
     * Export and upload the resulting image.
     */

    // apply the sharpening filter to the original image
    auto sharpened_image = A2 * v;
    // Reshape the sharpened image vector to a matrix
    auto sharpened_image_matrix = sharpened_image.reshaped<RowMajor>(height, width);
    // Save the sharpened image using stb_image_write
    const std::string sharpened_image_path = "../ch1_result/output_sharpened.png";
    if (stbi_write_png(sharpened_image_path.c_str(), width, height, 1,
                       convertToUnsignedChar(sharpened_image_matrix).data(), width) == 0) {
        PLOG_ERROR << "Could not save sharpened image";
        return 1;
    }
    PLOG_INFO << "Sharpened image saved to : " + sharpened_image_path;
    /**
     * Export the Eigen matrix A2 and vector w in the .mtx format.
     * Using a suitable iterative solver and preconditioner technique available in the LIS library
     * compute the approximate solution to the linear system A2x = w prescribing a tolerance of
     * 10−9. Report here the iteration count and the final residual.
     */
    exportMatrixMarketExtended(A2, w, "../ch1_result/A2_w.mtx");
    LIS_MATRIX A;
    LIS_VECTOR x, b;
    LIS_SOLVER solver;
    LIS_INT iter;
    LIS_REAL resid;
    double time;
    auto tol = 1.0e-9;

    // List of solvers to try
    std::vector<std::string> solvers = {"cg", "bicg", "jacobi", "gs", "bicgstab", "gmres"};

    LIS_DEBUG_FUNC_IN;

    lis_initialize(&argc, &argv);

    // Matrix and vectors setup
    lis_matrix_create(LIS_COMM_WORLD, &A);
    lis_vector_create(LIS_COMM_WORLD, &b);
    lis_vector_create(LIS_COMM_WORLD, &x);
    lis_matrix_set_type(A, LIS_MATRIX_CSR);

    const std::string input_file = "../ch1_result/A2_w.mtx";
    lis_input(A, b, x, const_cast<char *>(input_file.c_str()));
    lis_vector_duplicate(A, &x);

    for (const auto &solver_name: solvers) {
        lis_solver_create(&solver);
        // Set solver options dynamically based on the current solver_name
        std::string solver_option = std::format("-i {} -p ssor -tol {}", solver_name, tol);
        lis_solver_set_option(const_cast<char *>(solver_option.c_str()), solver);

        // Solve the system
        lis_solve(A, b, x, solver);

        // Get and log results
        lis_solver_get_iter(solver, &iter);
        lis_solver_get_time(solver, &time);
        lis_solver_get_residualnorm(solver, &resid);

        // Log details
        PLOG_INFO << "Solver: " + solver_name;
        PLOG_INFO << "Iterations: " + std::to_string(iter);
        PLOG_INFO << "Residual: " + std::format("{}", resid);
        PLOG_INFO << "Elapsed time: " + std::to_string(time) + " seconds";

        // Output results to .mtx and .png
        std::string output_file_mtx = "../ch1_result/" + solver_name + "_result.mtx";
        std::string output_file_png = "../ch1_result/" + solver_name + "_result.png";
        lis_output_vector(x, LIS_FMT_MM, const_cast<char *>(output_file_mtx.c_str()));

        /**
         * Import the previous approximate solution vector x in Eigen and then convert it into a .png image.
         * Upload the resulting file here
         */
        saveMatrixMarketToImage(output_file_mtx, output_file_png, height, width);

        // Clean up solver
        lis_solver_destroy(solver);
    }

    // Cleanup matrix and vectors
    lis_matrix_destroy(A);
    lis_vector_destroy(b);
    lis_vector_destroy(x);

    lis_finalize();

    /**
     * Write the convolution operation corresponding to the detection kernel Hlab as a matrix vector multiplication
     * by a matrix A3 having size mn × mn. Is matrix A3 symmetric?
     */
    // Define the detection kernel Hlab
    auto A3 = createLaplacianMatrixOptimized(height, width);
    PLOG_INFO << "The number of non-zero entries in A3 is: " + std::to_string(countNonZeroElements(A3));
    // Report if matrix A3 is symmetric
    PLOG_INFO << "Matrix A3 is symmetric: " + std::to_string(A3.isApprox(A3.transpose()));

    /**
     * Apply the previous edge detection filter to the original image by performing the matrix
     * vector multiplication A3v. Export and upload the resulting image.
     */
    // Apply the edge detection filter to the original image
    auto laplacian_image = A3 * v;
    // Reshape the edge detection image vector to a matrix
    auto laplacian_image_matrix = laplacian_image.reshaped<RowMajor>(height, width);
    // Save the edge detection image using stb_image_write
    const std::string laplacian_image_path = "../ch1_result/output_laplacian.png";
    if (stbi_write_png(laplacian_image_path.c_str(), width, height, 1,
                       convertToUnsignedChar(laplacian_image_matrix).data(), width) == 0) {
        PLOG_ERROR << "Could not save edge detection image";
        return 1;
    }
    PLOG_INFO << "Edge detection image saved to: " + laplacian_image_path;

    /**
     * Using a suitable iterative solver available in the Eigen library compute the approximate
     * solutionofthelinearsystem(I+A3)y= w,where I denotes the identity matrix,
     * prescribing a tolerance of 10−10. Report here the iteration count and the final residual.
     */
    // Define the identity matrix
    SparseMatrix<double> I(height * width, height * width);
    I.setIdentity();
    // Define the matrix A3
    auto A3_I = I + A3;
    // List of solver names and corresponding solver objects
    for (std::vector<std::string> solver_names = {"ConjugateGradient", "BiCGSTAB", "SparseLU"};
         const auto &solver_name: solver_names) {
        VectorXd y;

        if (solver_name == "ConjugateGradient") {
            // Conjugate Gradient solver
            ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
            cg.setTolerance(1.0e-10);
            cg.compute(A3_I);

            if (cg.info() != Eigen::Success) {
                PLOG_ERROR << "Solver: " + solver_name + " failed to converge";
                continue;
            }
            y = cg.solve(w);

            // Logging iteration count and residual
            PLOG_INFO << "Solver: " + solver_name;
            PLOG_INFO << "Iterations: " + std::to_string(cg.iterations());
            PLOG_INFO << "Final residual: " + std::format("{}", cg.error());

        } else if (solver_name == "BiCGSTAB") {
            // BiCGSTAB solver
            BiCGSTAB<SparseMatrix<double>> bicgstab;
            bicgstab.setTolerance(1.0e-10);
            bicgstab.compute(A3_I);

            if (bicgstab.info() != Success) {
                PLOG_ERROR << "Solver: " + solver_name + " failed to converge";
                continue;
            }
            y = bicgstab.solve(w);

            // Logging iteration count and residual
            PLOG_INFO << "Solver: " + solver_name;
            PLOG_INFO << "Iterations: " + std::to_string(bicgstab.iterations());
            PLOG_INFO << "Final residual: " + std::format("{}", bicgstab.error());

        } else if (solver_name == "SparseLU") {
            // SparseLU solver
            SparseLU<SparseMatrix<double>> sparse_lu;
            sparse_lu.compute(A3_I);

            if (sparse_lu.info() != Success) {
                PLOG_ERROR << "Solver: " + solver_name + " failed to converge";
                continue;
            }
            y = sparse_lu.solve(w);

            // the final residual for the linear system (I + A3)y = w
            // Logging iteration count and residual (SparseLU doesn't iterate)
            PLOG_INFO << "Solver: " + solver_name;
            PLOG_INFO << "Final residual: " + std::format("{}", (A3_I * y - w).norm());
        }

        // Reshape the solution vector to a matrix
        auto y_matrix = y.reshaped<RowMajor>(height, width);

        /**
         * Convert the image stored in the vector y into a .png image and upload it.
         */
        // Save the resulting image as a PNG file using stb_image_write
        const std::string y_image_path = "../ch1_result/" + solver_name + "_y.png";
        if (stbi_write_png(y_image_path.c_str(), width, height, 1, convertToUnsignedChar(y_matrix).data(), width) ==
            0) {
            PLOG_ERROR << "Could not save image for solver: " + solver_name;
            continue;
        }
        PLOG_INFO << std::string("Image saved for solver: ").append(solver_name).append(" to ").append(y_image_path);
    }

    /**
     * Comment the obtained results.
     */

    /*
    Compared to direct methods (LU factorization), the Jacobi method, and the Gauss-Seidel method, the improved BiCG and
    BiCGSTAB algorithms for the Conjugate Gradient Method have shown good performance in solving this large sparse
    matrix. Both methods reduced the residual to very small values, indicating good convergence. Overall, the BiCGSTAB
    algorithm converged the fastest compared to BiCG, though its accuracy was slightly lower. All iterative methods were
    able to converge, whereas the direct method was unsuitable for this system (no solution). In regard to the images
    obtained from the various filtering operations, it was observed that the first and last pixels of each row and
    column did not conform to what we initially expected. This is due to the fact that the neighbouring pixels are not
    available at the boundaries; therefore, zero padding is employed to extend the image, thus enabling the application
    of the kernel to the edge pixels. The presence of these zeros has an impact on the convolution calculation,
    resulting in the aforementioned effect. This can be mitigated by modifying the type of padding employed, for example
    extending the value of the image edge or using the wrap technique.
     */
    return 0;
}
