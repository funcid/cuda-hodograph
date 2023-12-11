#include "mex.hpp"
#include "mexAdapter.hpp"
#include <iostream>

using namespace matlab::data;
using matlab::mex::ArgumentList;
using namespace std;

class MexFunction : public matlab::mex::Function {
public:
    void operator()(ArgumentList outputs, ArgumentList inputs) {
    ArrayFactory factory;
    TypedArray<double> x = move(inputs[0]);
    const double z = x[0];
    outputs[0] = x;
    cout<<z<<endl;
    }
};