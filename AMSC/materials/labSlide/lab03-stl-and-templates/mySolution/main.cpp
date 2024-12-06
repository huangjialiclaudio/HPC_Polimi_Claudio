#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <chrono>
#include <map>
#include <unordered_map>
#include <tuple>
#include <memory>
#include <cassert>
#include <utility>
#include <functional>
#include <array>


// *use templates to make your matrix usable with different types
// *just like an std::vector can contain different elements, depending on
// *what you specified
template<typename T>
class SparseMatrix {
public:
  using Vector = std::vector<T>;
  // *constructor, takes no parameters but initializes: number of non-zero elements, number of rows and cols to zero
  SparseMatrix() : nrow(0) , ncol(0) , nnz(0) {};
  // *getter for number of rows
  size_t getNrow() const {return nrow;}
  // *getter for number of cols
  size_t getNcol() const {return ncol;}
  // *getter for number of number of non-zero
  size_t getNnz() const {return nnz;}

  // *print function: prints the number of rows, cols and nnz; moreover calls 
  void print(std::ostream& os = std::cout) const {
    os<< "nrows: " << nrow << " ncols: " << ncol << " nnz: " << nnz << std::endl;
    //_print, the virtual version of print that is specialized in the children class
    _print(os);
  }

  // *abstract virtual method vmult that implements vector multiplication
  virtual Vector vmult(const Vector& v) const = 0;
  // *abstract virtual method operator()(size_t i, size_t j) that implements element access in read only (const version)
  virtual const T& operator()(size_t i, size_t j) const = 0;
  // *abstract virtual method operator()(size_t i, size_t j) that implements element access in write (returns non-const reference)
  virtual T& operator()(size_t i, size_t j) = 0;
  // *virtual destructor
  virtual ~SparseMatrix() = default;

protected: // protected because need to be accessible to children!
  // *abstract virtual method _print that prints the matrix
  virtual void _print(std::ostream& os) const = 0;
  // *variable to store nnz
  // *variable to store number of rows
  // *variable to store number of cols
  size_t nnz, nrow, ncol;
};

template<typename T>
class MapMatrix : public SparseMatrix<T> {
public:
  using Vector = typename SparseMatrix<T>::Vector;
  virtual Vector vmult(const Vector& x) const override {
    // assert x.size() == m_ncols
    assert(x.size() == SparseMatrix::ncol);
    // allocate memory for result and initialize to 0
    Vector res(x.size());
    // loop over each element of the matrix and add contribute to result
    for(i = size_t; i < data.size(); i++){
      for(const auto& [j,v] : data[i]){
        res[i] += x[j] * v;
      }
    }
    return res;
  }

  virtual double& operator()(size_t i, size_t j) override {
    // check if we have enough rows, if not add them
    // find column entry, if not present add it
    // return value reference
  }
  virtual const double& operator()(size_t i, size_t j) const override {
    // return value reference with no check, we use the c++ convention of no bounds check 
  }
  virtual ~MapMatrix() override = default;
protected:
  virtual void _print(std::ostream& os) const {
    // print the data
  }

  std::vector<std::map<size_t, T>> data;
};

int main() {
  MapMatrix<double> m;
  m(0, 0) = 1;
  m(1, 1) = 1;
  m.print(std::cout);
  const auto x = m.vmult({{2, 3}});
  std::cout << x[0] << " " << x[1] << std::endl;
  return 0;
}