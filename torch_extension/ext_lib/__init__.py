# flake8: noqa
import os
import sys

# add current dir into the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir)))

# import library
import torch_extension_ops
