# Direct solvers (SparseLU and SparseLDLT) in Eigen

The goal of this lab is to test the sparse direct solvers available in the Eigen library and compare them with respect to the iterative methods we implemented in the previous labs. 

The native Eigen direct solvers are based on the Choleski and LU factorization in the case in which the input matrix is symmetric or not, respectively. The syntax for calling the `SparseLU` and `SimplicialLDLT` Eigen functions is similar to the one used for the iterative solvers `ConjugateGradient` and `BICGSTAB`:

```cpp
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;    // define solver
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                     // sanity check
      std::cout  << "cannot factorize the matrix" << std::endl; 
      return 0;
  }
  x = solvelu.solve(b);                                    // solve
```


## Exercise 1

Let us consider the Hilbert matrix of size $n=100$ defined as follows:
$$
H_{i,j} = \frac1{i+j+1} \quad \forall 1\le i,j\le 100.
$$

We want to compare the `SparseLU` and `SimplicialLDLT` direct solvers with respect to the hand-made Conjugate Gradient method for solving the linear system $Ax = b$, where $b$ is obtained by taking $x = (1,1,\ldots, 1)$ as exact solution.

Since the conditioning number of the Hilbert matrix is very large, we observe that the direct solvers does not work. The resulting error is of order $10^3$.
Instead, the CG method is able to compute a good approximation in only 13 iterations.

```cpp
#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "cg.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 100;
  SpMat A(n,n);                       // define Hilbert matrix
  for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++){
          A.coeffRef(i, j) = 1.0/(i+j+1);
      }
  }

  double tol = 1.e-8;                  // Convergence tolerance
  int result, maxit = 1000;            // Maximum iterations for CG
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<std::endl;
  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen Choleski direct solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;  // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; // decomposition failed
      return 0;
  }

  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Then with Eigen SparseLU solver
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;        // LU factorization
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                                         // solving
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Finally with hand-made CG
  x=0*x;
  result = CG(A, x, b, D, maxit, tol);                     // Call CG function
  std::cout << "Solution with Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}
```

## Exercise 2

In this exercise we consider a symmetric square matrix of size $n=1000$ obtained as the sum of the tridiagonal matrix (arising when discretizing a 1D laplacian using finite differences) and an Hilbert matrix of size $20\times 20$, namely $A = B + H$ with
$$
B_{i,i} = 2,\quad B_{i,i+1} = B_{i,i-1} = -1, \quad B_{i,j} = 0 \quad \forall j\neq i-1, i, i+1, \forall 1\le i\le n,
$$
$$
\text{and} \quad H_{i,j} = \frac1{i+j+1} \quad \forall 1\le i,j\le 20.
$$

We want to compare the `SparseLU` and `SimplicialLDLT` direct solvers with respect to the hand-made Conjugate Gradient method for solving the linear system $Ax = b$, where $b$ is obtained by taking $x = (1,1,\ldots, 1)$ as exact solution.

We observe that in this case the direct solvers works better than the CG method, that requires almost 1000 of iterations to converge and returns a larger error.

```cpp
#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "cg.hpp"

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double>;
  using SpVec=Eigen::VectorXd;

  int n = 1000;
  SpMat A(n,n);                       // define Hilbert matrix
  for (int i=0; i<20; i++) {
      for (int j=0; j<20; j++){
          A.coeffRef(i, j) = 1.0/(i+j+1);
      }
  }
  for (int i=0; i<n; i++) {
      A.coeffRef(i, i) += 2.0;
      if (i < n-1) A.coeffRef(i, i+1) += -1.0;
      if (i > 0) A.coeffRef(i,i-1) += -1.0;
  }

  double tol = 1.e-9;                 // Convergence tolerance
  int result, maxit = 999;            // Maximum iterations for CG
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<std::endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen Choleski direct solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;  // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;
      return 0;
  }
  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Then with Eigen SparseLU solver
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;        // LU factorization
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                                         // solving
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Now with hand-made CG
  x=0*x;
  result = CG(A, x, b, D, maxit, tol);              // Call CG function
  std::cout << "Solution with Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}
```

## Exercise 3

Download the matrices `navierstokes_mat.mtx` and `navierstokes_sym.mtx` from the webeep folder "Codes - Labs". 

We want to compare the `SparseLU` and `SimplicialLDLT` direct solvers with respect to the hand-made `cg` and `cgs` iterative methods for the linear system $Ax = b$, where $A$ corresponds to the downloaded matrices, while $b$ is obtained by taking $x = (1,1,\ldots, 1)$ as exact solution.

To avoid multiple compiling for changes in the input matrix, we provide the matrix file name as an input the the main program.

We observe that all the methods works when tested with the `navierstokes_sym.mtx` and both the $LU$ and $LDL^{\rm T}$ give a good precision. On the other hand, the `SimplicialLDLT` Eigen function does not work for the linear system defined by the non-symmetric `navierstokes_mat.mtx`. Indeed, as expected from the theory, the Choleski factorization works fine only for SPD matrices.

```cpp
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <iostream>
#include <string>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>

#include "cg.hpp"
#include "cgs.hpp"

int main(int argc, char *argv[]){
  using namespace Eigen;
  using namespace LinearAlgebra;

  if(argc != 2)
    {
      std::cerr << " Usage: provide matrix filename" << std::endl;
      return 1;
    }
  std::string matrixFile(argv[1]);
  // Some useful alias
  using SpMat = Eigen::SparseMatrix<double>;
  using SpVec = Eigen::VectorXd;

  // Read matrix
  SpMat A;
  Eigen::loadMarket(A, matrixFile);
  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;

  double tol = 1.e-8;                  // Convergence tolerance
  int result, maxit = 2000;            // Maximum iterations for CGS
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout<<"Norm of A-A.t: "<<B.norm()<<std::endl;

  // Create Rhs b
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A*e;
  SpVec x(A.rows());
  Eigen::DiagonalPreconditioner<double> D(A);

  // First with Eigen Choleski direct solver
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;   // LDLT factorization
  solver.compute(A);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;
      return 0;
  }
  x = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Then with Eigen SparseLU solver
  Eigen::SparseLU<Eigen::SparseMatrix<double> > solvelu;        // LU factorization
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;  // decomposition failed
      return 0;
  }

  x = solvelu.solve(b);                                         // solving
  std::cout << "Solution with Eigen LU:" << std::endl;
  std::cout << "effective error: "<<(x-e).norm()<< std::endl;

  // Now with hand-made CGS method
  x=0*x;
  // result = CG(A,x,b,D,maxit,tol);
  result = CGS(A, x, b, D, maxit, tol);          // Call CG - CGS function
  std::cout << "Solution with (Squared) Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "Error norm: "<<(x-e).norm()<< std::endl;

  return result;
}
```

## Exercise 4

In this exercise we construct block preconditioner for iterative solvers in Eigen. The preconditioned is built by using direct solvers (such as LU, QR, or Choleski factorization) to approximate the inverse of a block of the matrix $A$ defining the linear system $Ax = b$, where $b = (1,1,\ldots, 1)$.
```cpp
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <iostream>
#include <string>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include "cg.hpp"

int main(int argc, char *argv[]){
  using namespace Eigen;
  using namespace LinearAlgebra;

  if(argc != 2)
    {
      std::cerr << " Usage: provide matrix filename" << std::endl;
      return 1;
    }
  std::string matrixFileA(argv[1]);
  // Some useful alias
  using SpMat = Eigen::SparseMatrix<double>;
  using SpVec = Eigen::VectorXd;

  // Read matrix and check properties
  SpMat A;
  Eigen::loadMarket(A, matrixFileA);
  std::cout<<"Matrix size:"<<A.rows()<<"X"<<A.cols()<<std::endl;
  std::cout<<"Non zero entries:"<<A.nonZeros()<<std::endl;
  SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
  std::cout << "Norm of A-A.t: " << B.norm() << std::endl << std::endl;

  // Create Rhs b
  SpVec b = SpVec::Ones(A.rows());
  SpVec x(A.rows());
  SpVec y(A.rows());

  // Solution with hand-made CG method
  double tol = 1.e-10;                  // Convergence tolerance
  int result, maxit = 1000;             // Maximum iterations

  MatrixXd I = MatrixXd::Identity(A.rows(),A.cols());
  Eigen::DiagonalPreconditioner<double> Id(I); // preconditioner
  result = CG(A,x,b,Id,maxit,tol);
  std::cout << "Solution with Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl << std::endl;

  // Compute block preconditioner with Eigen Choleski direct solver
  int m = 200;
  I.topLeftCorner(A.rows()-m, A.cols()-m) = A.topLeftCorner(A.rows()-m,A.cols()-m);
  SpMat P = I.sparseView();

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(P);
  if(solver.info()!=Eigen::Success) {                          // sanity check
      std::cout << "cannot factorize the matrix" << std::endl;
      return 0;
  }
  y = solver.solve(b);                                         // solving
  std::cout << "Solution with Eigen Choleski:" << std::endl;
  std::cout << "relative residual: "<<(P*y-b).norm()<< std::endl << std::endl;

  // Solution with preconditioned CG method
  SpVec x2(A.rows()); tol = 1.e-10; maxit = 1000;
  result = CG(A,x2,b,solver,maxit,tol);
  std::cout << "Solution with precond Conjugate Gradient:" << std::endl;
  std::cout << "iterations performed: " << maxit << std::endl;
  std::cout << "tolerance achieved  : " << tol << std::endl;
  std::cout << "comparison solutions : " << (x-x2).norm() << std::endl;

  return result;
}
```

# Domain Decomposition preconditioners

In this lab we want to explore the Additive Schwarz and Algebraic Multigrid methods implemented in the LIS library in order to define advanced preconditioners for sparse linear systems.

## Additive Schwarz preconditioner

Domain decomposition (DD) methods are techniques based on a decomposition of the spatial domain of the problem into several subdomains. Such reformulations are usually motivated by the need to create solvers which are easily parallelized on coarse grain parallel computers, though sometimes they can also reduce the complexity of solvers on sequential computers.

In order to obtain a domain decomposition for the linear system $Ax = b$, one needs to decompose the unknowns in the vector $x$ into subsets. 
For the sake of simplicity, we consider only a two subdomain case. Let $\mathcal{N}_i$ , $i = 1,2$ be a partition of the indices corresponding to the vector $x$ and  $R_i$ , $i = 1,2$ denote the matrix that, when applied to the vector $x$, returns only those values associated with the indices in $N_i$. The matrices $R_i$ are often referred to as the restriction operators, while $R^{\rm T}_i$ are the interpolation matrixes.

One can define the restriction of the matrix $A$ to the first and second unknowns using the restriction matrices:
$$
A_j = R_j A R^{\rm T}_j,\quad j = 1, 2.
$$
Thus the matrix $A_j$ is simply the subblock of A associated with the indices in $\mathcal{N}_j$. The preconditioned system obtained by applying the Additive Schwarz method reads
$$
(R^{\rm T}_1 A^{-1}_1 R_1 + R^{\rm T}_2 A^{-1}_2 R_2) A x = (R^{\rm T}_1 A^{-1}_1 R_1 + R^{\rm T}_2 A^{-1}_2 R_2) b.
$$
Since the sizes of the matrices $A_1$ and $A_2$ are smaller than the original matrix $A$, the inverse matrices $A_1^{-1}$ and $A_2^{-1}$ are easier to compute. Using this preconditioner for a stationary iterative method yields
$$
x^{n+1} = x^n + (R^{\rm T}_1 A^{-1}_1 R_1 + R^{\rm T}_2 A^{-1}_2 R_2) (b- Ax^n).
$$

Even if the matrices $A_i$, $i=1,2$, are easier to be inverted, in LIS it is preferred not to compute the inverse matrices exactly. Another preconditioned technique must be used together with the Additive Schwarz method in order to define the matrices $B_i$ approximating $A^{-1}_i$. The corresponding preconditioner is computed as $R^{\rm T}_1 B_1 R_1 + R^{\rm T}_2 B_2 R_2$. We can check this by the following test:

```
mpirun -n 4 ./test1 testmat0.mtx 2 sol.mtx hist.txt -i cg

mpirun -n 4 ./test1 testmat0.mtx 2 sol.mtx hist.txt -i cg -adds true
```
We observe that the option `-adds true` needed to specify the use of the Additive Schwarz method does not provide any preconditioner. In both cases, we get the following result

```
number of processes = 4
matrix size = 100 x 100 (460 nonzero entries)

initial vector x      : all components set to 0
precision             : double
linear solver         : CG
preconditioner        : none
convergence condition : ||b-Ax||_2 <= 1.0e-12 * ||b-Ax_0||_2
matrix storage format : CSR
linear solver status  : normal end
CG: number of iterations = 15
```

If instead we write `mpirun -n 4 ./test1 testmat0.mtx 2 sol.mtx hist.txt -i cg -adds true -p jacobi`, we get
```
initial vector x      : all components set to 0
precision             : double
linear solver         : CG
preconditioner        : Jacobi + Additive Schwarz
convergence condition : ||b-Ax||_2 <= 1.0e-12 * ||b-Ax_0||_2
matrix storage format : CSR
linear solver status  : normal end

CG: number of iterations = 14
```

## Some examples:

Assess the performances of the Additive Schwarz preconditioner implemented in LIS on different sparse linear systems:

```
mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i gmres
mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i gmres -adds true -p ssor
```
In this case the number of iterations for reaching the desired tolerance reduces from 36 to 14! 

```
mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i bicgstab
mpirun -n 2 ./test1 testmat2.mtx 2 sol.mtx hist.txt -i bicgstab -adds true -p ilu -ilu_fill 2
```
In this case the number of iterations reduces from 36 to 6 and also the total elapsed time for the computation is reduced. 

We can also set the number of iterations of the Additive Schwarz method using the `adds_iter` option. The effect of this parameter can be observed in the following example.
```
mpirun -n 4 ./test2 100 100 0 sol.mtx hist.txt -i gmres -p ssor -adds true -adds_iter 2
mpirun -n 4 ./test2 100 100 0 sol.mtx hist.txt -i gmres -p ssor -adds true -adds_iter 3
mpirun -n 4 ./test2 100 100 0 sol.mtx hist.txt -i gmres -p ssor -adds true -adds_iter 4
```
The number of iterations drops from 104 to 81 (using `adds_iter = 3`) and 70 (using `adds_iter = 4`). However we can observe that the total time for the solution of the linear system is similar because the time required for computing the preconditioner increases.


## Exercise 1

Evaluate and compare the effect of the Additive Schwarz preconditioner for solving the linear systems deefined by the matrices [bcsstm12.mtx](https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstm12.tar.gz) and [nos1.mtx](https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos1.mtx.gz).

```
wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstm12.tar.gz
tar -xf bcsstm12.tar.gz
mv bcsstm12/bcsstm12.mtx .
rm -rf bcsstm12 bcsstm12.tar.gz 

wget https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos1.mtx.gz
gzip -dk nos1.mtx.gz 

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -maxiter 5000 -tol 1e-10

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p sainv 

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p sainv -adds true

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p sainv -adds true -adds_iter 2

mpirun -n 4 ./test1 bcsstm12.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p sainv -adds true -adds_iter 3
```
In this case we can see that the number of iterations required to convergence does not depend monotonically on the choice of `adds_iter`.

```
wget https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos1.mtx.gz
gzip -dk nos1.mtx.gz 

mpirun -n 4 ./test1 nos1.mtx 1 sol.txt hist.txt -i cg
mpirun -n 4 ./test1 nos1.mtx 1 sol.txt hist.txt -i cg -p ssor
mpirun -n 4 ./test1 nos1.mtx 1 sol.txt hist.txt -i cg -p ssor -adds true -adds_iter 2
mpirun -n 4 ./test1 nos1.mtx 1 sol.txt hist.txt -i cg -p jacobi -adds true -adds_iter 5
```


## Algebraic Multigrid preconditioner (extra)

Algebraic multigrid (AMG) solves linear systems based on multigrid principles, but in a way that only depends on the coefficients in the underlying matrix. The AMG method determines coarse “grids”, inter-grid transfer operators, and coarse-grid equations based solely on the matrix entries.

The AMG preconditioner implemented in LIS is called `SA-AMG` and can be specified as an option by typing `-p saamg`. Unfortunately this preconditioner is incompatible with the use of mpi for multi-processors computations. Therefore, a serial reconfiguration of the LIS installation is needed in order to test the AMG method. 


### Exercise (extra)

Assess the performances of the SA-AMG preconditioner on the matrices considered in the previous examples.

```
gfortran test1.c -I${mkLisInc} -L${mkLisLib} -llis -o test1
./test1 stokes_sym.mtx 2 sol.mtx hist.txt -i cg
./test1 stokes_sym.mtx 2 sol.mtx hist.txt -i cg -p saamg
./test1 stokes_sym.mtx 2 sol.mtx hist.txt -i cg -p saamg -adds true
./test1 stokes_sym.mtx 2 sol.mtx hist.txt -i cg -p saamg -adds true -adds_iter 4

./test1 testmat2.mtx 1 sol.mtx hist.txt -i gmres
./test1 testmat2.mtx 1 sol.mtx hist.txt -i gmres -p saamg
./test1 testmat2.mtx 1 sol.mtx hist.txt -i gmres -p saamg -saamg_unsym true
./test1 testmat2.mtx 1 sol.mtx hist.txt -i gmres -p saamg -adds true


gfortran test2.c -I${mkLisInc} -L${mkLisLib} -llis -o test2
./test2 100 100 1 sol.mtx hist.txt -i cg
./test2 100 100 1 sol.mtx hist.txt -i cg -p saamg
./test2 100 100 1 sol.mtx hist.txt -i bicgstab -p saamg
./test2 100 100 1 sol.mtx hist.txt -i bicgstab -p saamg -adds true
```