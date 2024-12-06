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

    int n = 100;
    SpMat A(n,n);
    for(int i = 0; i < n; i++){
        A.coeffRef(i,i) = -8;
        if(i > 0)   A.coeffRef(i,i-1) = 2;
        if(i < n-1)   A.coeffRef(i,i+1) = 4;
        if(i < n-2)   A.coeffRef(i,i+2) = 1;
    }
    cout<<A.norm()<<endl<<endl;

    saveMarket(A,"Aex2.mtx");
}