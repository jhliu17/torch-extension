cmake_minimum_required(VERSION 3.0.0)
project(torch_ext VERSION 0.1.0)

include(CTest)
enable_testing()

# conda envs
set(CONDA_ENVS_PATH "/Users/junhao/anaconda3/envs/torch")

# find pytorch
set(CMAKE_PREFIX_PATH "${CONDA_ENVS_PATH}/lib/python3.8/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${CONDA_ENVS_PATH}/lib/python3.8/site-packages/torch/lib")

# find python inside envs
set(PYTHON_LIBRARY "${CONDA_ENVS_PATH}/lib/libpython3.8.dylib")
set(PYTHON_INCLUDE_DIR "${CONDA_ENVS_PATH}/include/python3.8")
set(PYTHON_EXECUTABLE "${CONDA_ENVS_PATH}/bin/python3.8")

# find pybind11
set(CMAKE_PREFIX_PATH "${CONDA_ENVS_PATH}/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# set target
pybind11_add_module(torch_ext torch_ext.cpp)
set_target_properties(
    torch_ext PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)
target_include_directories(
    torch_ext
    PRIVATE ${TORCH_API_PATH}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(
    torch_ext
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
    PRIVATE ${pybind11_LIBRARIES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
