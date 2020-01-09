from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='fwi',
      ext_modules=[cpp_extension.CppExtension('fwi', ['Torch_Fwi.cpp', 'Parameter.cpp', 'libCUFD.cu', 'el_stress.cu', 'el_velocity.cu', 'el_stress_adj.cu', 'el_velocity_adj.cu', 'Model.cu', 'Cpml.cu', 'utilities.cu',	'Src_Rec.cu', 'Boundary.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
