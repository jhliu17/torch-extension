#include <cassert>
#include <cstddef>

#include <ATen/core/TensorBody.h>
#include <pybind11/detail/common.h>
#include "torch/library.h"
#include "torch/extension.h"

/**
 * @brief add two tensor and save to output
 * 
 * @param out output tensor space
 * @param a input tensor
 * @param b input tensor
 */
void add_two(torch::Tensor &out, torch::Tensor &a, torch::Tensor &b) {
  float *out_p = (float *)(out.data_ptr());
  float *a_p = (float *)a.data_ptr();
  float *b_p = (float *)b.data_ptr();
  
  // check shape
  assert(a.sizes() == b.sizes());
  for (size_t i = 0; i < out.numel(); ++i) {
    out_p[i] = a_p[i] + b_p[i];
  }
}

/**
 * @brief sub two tensor and save to output
 * 
 * @param out output tensor space
 * @param a input tensor
 * @param b input tensor
 */
void sub_two(torch::Tensor &out, torch::Tensor &a, torch::Tensor &b) {
  float *out_p = (float *)(out.data_ptr());
  float *a_p = (float *)a.data_ptr();
  float *b_p = (float *)b.data_ptr();

  // check shape
  assert(a.sizes() == b.sizes());
  for (size_t i = 0; i < out.numel(); ++i) {
    out_p[i] = a_p[i] - b_p[i];
  }
}

PYBIND11_MODULE(torch_extension_ops, m) {
  m.def("add", &add_two, "cpp kernel: add two variables");
  m.def("sub", &sub_two, "cpp kernel: sub two variables");
}
