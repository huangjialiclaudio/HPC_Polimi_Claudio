#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <string>

using namespace Eigen;

int main(int argc, char* argv[]){

    using SpMat = SparseMatrix<double>;
    using SpVec = VectorXd;

    SpMat A,B,C;
    loadMarket(A,"matA.mtx");
    loadMarket(B,"matA.mtx");
    loadMarket(C,"matA.mtx");

    MatrixXd M = MatrixXd::Zero(A.rows()+C.rows(),A.cols()+C.cols());
    M.topLeftCorner(A.rows(),A.cols()) = A;
    M.topRightCorner(C.rows(),C.cols()) = B.transpose();
    M.bottomLeftCorner(C.rows(),C.cols()) = B;
    M.bottomRightCorner(A.rows(),A.cols()) = C;
    SpMat MM = M.sparseView();

    std::cout<<"Matrix Size of M: "<<MM.rows()<<"X"<<MM.cols()<<std::endl;
    std::cout<<"Norm of M: "<<MM.norm()<<std::endl;
    SpMat SK = SpMat(MM.transpose()) - M;
    std::cout<<"Norm of M-M^t: "<<SK.norm()<<std::endl;
    
    // SpVec b = VectorXd::Ones(MM.rows());
    // SpVec x(MM.rows());
    // SparseLU<SparseMatrix<double>> solverLU;
    // solverLU.compute(MM);
    // if(solverLU.info() != Success) {
    //     std::cout << "cannot factorize the matrix" << std::endl;
    //     return 0;
    // }
    // x = solverLU.solve(b);
    // std::cout << "Solution with LU complete system:" << std::endl;
    // std::cout << "absolute residual: " << (b-M*x).norm() << std::endl;
    

    int n = 100;
    SpMat D(n,n);
    for(int i = 0; i < n; i++){
        D.coeffRef(i,i) = 8;
        if(i > 0)   D.coeffRef(i,i-1) = -2;
        if(i < n-1)   D.coeffRef(i,i+1) = -4;
        if(i < n-2)   D.coeffRef(i,i+2) = -1;
    }
    std::cout << "Norm of D: " << D.norm() << std::endl;

    MatrixXd Af;
    Af = MatrixXd(D);
    EigenSolver<MatrixXd> eigensolver(Af);
    if(eigensolver.info() != Success) {
        std::cout << "cannot factorize the matrix" << std::endl;
        return 0;
    }
    std::cout << "the eigenvalues of D are: " << eigensolver.eigenvalues() << std::endl;

    std::string matrixFileOut("./Aex2.mtx");
    saveMarket(A, matrixFileOut);
    
}