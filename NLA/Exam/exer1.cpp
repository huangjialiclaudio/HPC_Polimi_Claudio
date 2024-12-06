#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <string>

using namespace Eigen;
using namespace std;

int main(int argc, char* argv[]){

    using SpMat = SparseMatrix<double>;
    using SpVec = VectorXd;

    SpMat A;
    loadMarket(A,"A_exam1.mtx");
    cout << "The size of Matrix A is:" << A.rows()<<"X"<<A.cols() << endl;
    cout << "The norm of A is:" << A.norm() << endl << endl;

    VectorXd x = VectorXd::Ones(A.cols());
    VectorXd b = A*x;
    cout << "The norm of b is:" << b.norm() << endl << endl;

    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solverQR;
    solverQR.compute(A);
    if(solverQR.info()!=Eigen::Success) {                   
        std::cout << "cannot factorize the matrix" << std::endl;
        return 0;
    }
    VectorXd xQR = solverQR.solve(b);
    cout << "The error norm of XQR-X is:" << (xQR-x).norm() << endl << endl;

    Eigen::ConjugateGradient<SpMat, Lower|Upper> cg;
    cg.setTolerance(1e-8);
    cg.compute(A.transpose()*A);
    VectorXd xEND = cg.solve(A.transpose()*b);
    cout << "Eigen CG: " << endl;
    cout << "#iterations:     " << cg.iterations() << endl;
    cout << "effective error: "<< (xEND-x).norm() << endl << endl;

    cout << "xQR - Xend error: "<< (xQR-xEND).norm() << endl << endl;
    
    int n = A.cols();
    int m = A.rows();
    MatrixXd A1(n,n),A2(n,n);
    A1 = A.topLeftCorner(n,n);
    A2 = A.bottomLeftCorner(m-n,n);
    VectorXd b1(n),b2(n);
    b2 = b.tail(n);
    cout << "A2 norm is: "<< A1.norm() << endl;
    cout << "b2 norm is: "<< b2.norm() << endl << endl;

}