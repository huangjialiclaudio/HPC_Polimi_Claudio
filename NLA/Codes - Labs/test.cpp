#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

int main(){
    int n = 10;
    MatrixXd m = MatrixXd::Random();
    cout << m.reshaped(2, 8);
    cout<<m;
    // int index;
    // for(int i = 0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         index = i * n +j;
    //         v(index) = m(i,j);
    //     }
    // }
    
    return 0;
}