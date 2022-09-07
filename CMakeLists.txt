cmake_minimum_required(VERSION 3.0.0)
project(torch_extension VERSION 0.1.0)

# set conda envs
set(CONDA_ENVS_PATH "$ENV{HOME}/anaconda3/envs/torch")

# find pytorch inside envs
find_package(Torch REQUIRED PATHS "${CONDA_ENVS_PATH}/lib/python3.8/site-packages/torch/share/cmake/Torch")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${CONDA_ENVS_PATH}/lib/python3.8/site-packages/torch/lib")

# find python inside envs
set(PYTHON_LIBRARY "${CONDA_ENVS_PATH}/lib/libpython3.8.dylib")
set(PYTHON_INCLUDE_DIR "${CONDA_ENVS_PATH}/include/python3.8")
set(PYTHON_EXECUTABLE "${CONDA_ENVS_PATH}/bin/python3.8")

# find pybind11
find_package(pybind11 REQUIRED PATHS "${CONDA_ENVS_PATH}/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")

# set pybind11 target
pybind11_add_module(torch_extension_ops torch_extension/src/ops.cpp)
set_target_properties(
    torch_extension_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/torch_extension/ext_lib"
)
target_include_directories(
    torch_extension_ops
    PRIVATE ${TORCH_INCLUDE_DIRS}
    PRIVATE ${pybind11_INCLUDE_DIRS}
)
target_link_libraries(
    torch_extension_ops
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
    PRIVATE ${pybind11_LIBRARIES}
)
