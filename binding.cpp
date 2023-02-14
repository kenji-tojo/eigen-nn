#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "eignn/module.hpp"


namespace nb = nanobind;
using namespace nb::literals;


NB_MODULE(eignn, m) {

    m.def("inspect_tensor", [](
            nb::tensor<float, nb::shape<nb::any, nb::any>> &tensor
    ) {
        using namespace std;
        cout << "tensor shape: " << tensor.shape(0) << "x" << tensor.shape(1) << endl;
    });

}