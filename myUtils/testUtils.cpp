//
// Created by Claudio on 2024/12/14.
//
#include <iostream>
#include "ClaudioUtils.h"

using namespace std;
using namespace ClaudioUtils;

int main(){
    int a = 4, b = 3;
    int c = ubound(a,b);
    double a1=0.17, b1=0.5;
    double c1 = ubound(a1,b1);
    cout << c << endl;
    cout << c1 << endl;
    return 0;
}