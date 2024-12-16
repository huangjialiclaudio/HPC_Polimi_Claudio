#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>


using namespace std;
using namespace Eigen;

// define QR class
class QR {
public:
    //construct function
    QR(MatrixXd A) : A(A) {};
    //compute matrixQ function
    MatrixXd getMatrixQ() const { 
        HouseholderQR<MatrixXd> qr(A);
        MatrixXd Q = qr.householderQ(); 
        return Q;
    }
    //deconstruct function
    ~QR() = default;
private:
    MatrixXd A;
};

//define powerSVD class
class PowerSVD {
public:
    //construct function
    PowerSVD(MatrixXd A) : A(A) { m = A.rows(); n = A.cols();};
    //compute matrixU function
    MatrixXd getMatrixU() const { 
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        return svd.matrixU();
    }
    //compute matrixV function
    MatrixXd getMatrixV() const { 
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        return svd.matrixV();
    }
    //compute matrixSigma function
    MatrixXd getMatrixSigma() const {
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        return svd.singularValues().asDiagonal();
    }
    //deconstruct function
    ~PowerSVD() = default;
protected:
    MatrixXd A;
    int m,n;
};

//rSVD extend from PowerSVD
class rSVD : public PowerSVD {
public:
    //construct function
    rSVD(MatrixXd A, int k) : PowerSVD(A), k(k){};
    void rotate() {
        //create random matrix
        MatrixXd Omega = MatrixXd::Random(n, k); 
        MatrixXd Y = A * Omega;

        //QR decomposition
        QR qr(Y);
        MatrixXd Q = qr.getMatrixQ();

        //rotate
        A = Q * A;
        return;
    }
    //deconstruct function
    ~rSVD() = default;
private:
    //col of random matrix
    int k;
};


int main() {
    // Set the size of the matrix A
    int m = 100, n = 50, k = 10;

    // Initialize matrix A with random values
    MatrixXd A = MatrixXd::Random(m, n);

    rSVD rsvd(A,k);
    rsvd.rotate();

    cout << "U (Left Singular Vectors):" << endl << rsvd.getMatrixU().norm() << endl << endl;
    cout << "Sigma (Singular Values):" << endl << rsvd.getMatrixSigma().norm() << endl << endl;
    cout << "Vt (Right Singular Vectors):" << endl << rsvd.getMatrixV().transpose().norm() << endl << endl;
    return 0;
}