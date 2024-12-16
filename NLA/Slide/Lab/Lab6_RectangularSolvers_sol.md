# Least square solvers (QR and LSCG) for rectangular systems in Eigen

In this lab we aim at exploring the sparse solvers available in the Eigen library for the solution of rectangular linear systems. 

The native Eigen direct solver for least square rectangular problem is based on the QR factorization. The `SparseQR` Eigen class implements a left-looking QR decomposition with numerical column pivoting. When a column has a norm less than a given tolerance it is implicitly permuted to the end. The QR factorization obtained is given by $A P = Q R$ where $R$ is upper triangular or trapezoidal. $P$ is the column permutation which is the product of the fill-reducing and the numerical permutations and $Q$ is the orthogonal matrix represented as products of Householder reflectors.

The syntax for calling the `SparseQR` Eigen functions is similar to the one used for the other sparse solvers :
```
  Eigen::SparseQR<Eigen::SparseMatrix<double>, COLAMDOrdering<int>> solver;   
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                     // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; 
      return 0;
  }
  x = solver.solve(b);                                    // solve
```

## Direct solvers for dense matrices

Other alternatives are available for dense matrix (see [EigenDense module](https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html)). The most general and accurate direct method to solve under- or over-determined linear systems in the least squares sense, is the SVD decomposition. Eigen provides two implementations. The recommended one is the `BDCSVD` class, which scales well for large problems and automatically falls back to the `JacobiSVD` class for smaller problems. 

Another possibility which is usually faster than SVD and about as accurate, is `CompleteOrthogonalDecomposition`. If you know more about the problem, the table above contains methods that are potentially faster. If your matrix is full rank, `HouseHolderQR` is the method of choice.

## Iterative solvers

In Eigen there is also an iterative solver for sparse least-square problems. It is given by the Least Square Conjugate Gradient Method applied in the contest of rectangular systems. 

The `Eigen::LeastSquaresConjugateGradient` solves for the least-squares solution to $A x = b$ using an iterative conjugate gradient algorithm on the matrix $A^{T} A$. The matrix $A$ can be non symmetric and rectangular, but the matrix $A^{T} A$ should be positive-definite to guaranty stability.


### Exercise 1

We want to compare the `SparseQR` direct solver with respect to the `LeastSquaresConjugateGradient` for the approximate solution of the rectangular system $Ax = b$, where $b$ is obtained by taking $x = (1,1,\ldots, 1)$ as exact solution. We consider some rectangular matrices available in the matrix market website.

```
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <iostream>
#include <string>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>

int main(int argc, char *argv[]){
  using namespace Eigen;

  if(argc != 2)
    {
      std::cerr << " Usage: provide matrix filename" << std::endl;
      return 1;
    }
  std::string matrixFile(argv[1]);

  // Some useful alias
  using SpMat = SparseMatrix<double>;
  using SpVec = VectorXd;

  // Read matrix
  SpMat A;
  loadMarket(A, matrixFile);

  // Create Rhs b
  SpVec e = SpVec::Ones(A.cols());
  SpVec b = A*e;
  SpVec x(A.cols());

  // solve with Eigen QR factorization
  Eigen::SparseQR<Eigen::SparseMatrix<double>, COLAMDOrdering<int>> solver;
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {
      std::cout << "cannot factorize the matrix" << std::endl;
      return 0;
  }
  x = solver.solve(b);
  std::cout << "Solution with Eigen QR:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // solve with Eigen LeastSquareConjugateGradient solver
  LeastSquaresConjugateGradient<SparseMatrix<double> > lscg;
  lscg.compute(A);
  x = lscg.solve(b);
  std::cout << "Solution with Eigen LSCG:" << std::endl;
  std::cout << "#iterations:     " << lscg.iterations() << std::endl;
  std::cout << "relative residual: " << lscg.error()      << std::endl;
  return 1;
}
```

# Singular value decomposition (SVD) in Eigen

SVD decomposition consists in decomposing any n-by-p matrix $A$ as a product
$$
A = U S V^T.
$$

where $U$ is a n-by-n unitary, $V$ is a p-by-p unitary, and $S$ is a n-by-p real positive matrix which is zero outside of its main diagonal; the diagonal entries of $S$ are known as the singular values of $A$ and the columns of $U$ and $V$ are known as the left and right singular vectors of $A$ respectively.

Singular values are always sorted in decreasing order.

You can ask for only thin $U$ or $V$ to be computed, meaning the following. In case of a rectangular n-by-p matrix, letting m be the smaller value among n and p, there are only m singular vectors; the remaining columns of $U$ and $V$ do not correspond to actual singular vectors. Asking for thin $U$ or $V$ means asking for only their m first columns to be formed. So $U$ is then a n-by-m matrix, and $V$ is then a p-by-m matrix. Notice that thin $U$ and $V$ are all you need for (least squares) solving.

The `solve()` method in the BDCSVD class can be directly used to solve linear squares systems. It is not enough to compute only the singular values (the default for this class); you also need the singular vectors but the thin SVD decomposition suffices for computing least squares solutions.

### Exercise 2

- Compute the singular values of the matrix well341.mtx available on webeep

- check that the computed singular value decomposition is correct

```
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <string>
#include <unsupported/Eigen/SparseExtra>

int main(int argc, char *argv[]){
  using namespace Eigen;

  if(argc != 2)
    {
      std::cerr << " Usage: provide matrix filename" << std::endl;
      return 1;
    }
  std::string matrixFile(argv[1]);

  // Some useful alias
  using SpMat = SparseMatrix<double>;

  // Read matrix
  SpMat A;
  loadMarket(A, matrixFile);
  MatrixXd Ad = MatrixXd(A); // convert to dense for SVD

  // compute Eigen SVD
  Eigen::BDCSVD<Eigen::MatrixXd> svd (Ad, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorXd W = svd.singularValues();
  std::cout << "singular values: " << W << "\n";
  MatrixXd Asvd = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
  MatrixXd diff = Asvd - Ad;
  std::cout << "diff:\n" << diff.norm() << "\n";
  return 1;
}
```

### Exercise 3 

We want to compare the `SparseQR` direct solver with respect to the `BDCSVD` solver for the approximate solution of the rectangular system $Ax = b$, where $b$ is obtained by taking $x = (1,1,\ldots, 1)$ as exact solution.

 - Consider the rectangular matrices used for Exercise 1.
 
 ```
 #include <Eigen/Core>
#include <Eigen/SparseQR>
#include <Eigen/SVD>
#include <iostream>
#include <string>
#include <unsupported/Eigen/SparseExtra>

int main(int argc, char *argv[]){
  using namespace Eigen;

  if(argc != 2)
    {
      std::cerr << " Usage: provide matrix filename" << std::endl;
      return 1;
    }
  std::string matrixFile(argv[1]);

  // Some useful alias
  using SpMat = SparseMatrix<double>;
  using SpVec = VectorXd;

  // Read matrix
  SpMat A;
  loadMarket(A, matrixFile);
  MatrixXd Ad = MatrixXd(A); // convert to dense for SVD

  // Create Rhs b
  std::cout << "Size of A: " << A.rows() << " x " << A.cols() << std::endl;
  SpVec e = SpVec::Ones(A.cols());
  SpVec b = A*e;
  SpVec x(A.cols());

  // solve with Eigen QR factorization
  Eigen::SparseQR<Eigen::SparseMatrix<double>, COLAMDOrdering<int>> solver;
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {
      std::cout << "cannot factorize the matrix" << std::endl;
      return 0;
  }
  x = solver.solve(b);
  std::cout << "Solution with Eigen QR:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // solve with Eigen SVD
  Eigen::BDCSVD<Eigen::MatrixXd> svd (Ad, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorXd W = svd.singularValues();
  std::cout << "singular values: " << W << "\n";
  MatrixXd Asvd = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
  MatrixXd diff = Asvd - Ad;
  std::cout << "diff:\n" << diff.norm() << "\n";

  x = svd.solve(b);
  std::cout << "Solution with Eigen SVD:" << std::endl;
  std::cout << "absolute error: " << (x-e).norm()  << std::endl;
  return 1;
}
```

### Exercise 4

In this exercise we consider a symmetric, tridiagonal matrix of size $n=10000$.

We want to compare the `CG` and `BiCGSTAB` iterative solvers for the approximate solution of $Ax = b$, where $b$ is obtained by taking $x = (1,1,\ldots, 1)$ as exact solution.

```
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
  cout << "relative residual: " << cg.error()      << endl;
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
```