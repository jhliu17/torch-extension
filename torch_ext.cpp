#include <ATen/core/TensorBody.h>
#include <cstdint>
#include <iostream>
#include <pybind11/detail/common.h>
#include <torch/library.h>
#include "torch/extension.h"

void add_two(
    torch::Tensor& out,
    torch::Tensor& a,
    torch::Tensor& b
){
    float* out_p = (float*)(out.data_ptr());
    float* a_p = (float*)a.data_ptr();
    float* b_p = (float*)b.data_ptr();
    for (std::int64_t i = 0; i < out.numel(); ++i) {
        out_p[i] = a_p[i] + b_p[i];
    }
}


PYBIND11_MODULE(torch_ext, m) {
    m.def("torch_add_two", &add_two, "custom add two cpp kernel");
}

// TORCH_LIBRARY(add2, m) {
//     m.def("torch_add_two", add_two);
// }
