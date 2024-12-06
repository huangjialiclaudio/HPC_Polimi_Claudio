#include <cstdlib>                      
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include "bcgstab.hpp"

using std::endl;
using std::cout;

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 10000;
  SpMat A(n,n);                       // define matrix
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) = i+1;
      if(i>0) A.coeffRef(i, i-1) = -1.0;
      if(i<n-1) A.coeffRef(i, i+1) = -1.0;
  }

  double tol = 1.e-10;                // Convergence tolerance
  int result, maxit = 1000;           // Maximum iterations

  cout << "Matrix size:" << A.rows() << "X" << A.cols() << endl;
  cout << "Non zero entries:" << A.nonZeros() << endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());    // define exact sol
  // for (int i=0; i<n/2; i++) e[2*i] = -1.0;
  SpVec b = A*e;
  cout << "norm of the rhs: "<< b.norm() << endl;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen CG
  Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
  cg.setMaxIterations(maxit);
  cg.setTolerance(tol);
  cg.compute(A);
  x = cg.solve(b);
  cout << "Eigen CG: " << endl;
  cout << "#iterations:     " << cg.iterations() << endl;
  cout << "estimated error: " << cg.error()      << endl;
  cout << "effective error: "<< (x-e).norm() << endl;

  // Then with BICGSTAB
  x=0*x;
  result = BiCGSTAB(A, x, b, D, maxit, tol);   
  cout << "BiCGSTAB: " << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;
  cout << "Error norm: "<<(x-e).norm()<< endl;

  return result;
}
