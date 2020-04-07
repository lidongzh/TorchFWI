from torch.utils.cpp_extension import load

def load_fwi(path):
    fwi = load(name="fwi",
            sources=[path+'/Torch_Fwi.cpp', path+'/Parameter.cpp', path+'/libCUFD.cu', path+'/el_stress.cu', path+'/el_velocity.cu', path+'/el_stress_adj.cu', path+'/el_velocity_adj.cu', path+'/Model.cu', path+'/Cpml.cu', path+'/utilities.cu',	path+'/Src_Rec.cu', path+'/Boundary.cu'],
            extra_cflags=[
                '-O3'
            ],
            extra_include_paths=['/usr/local/cuda/include', path+'/rapidjson'],
            extra_ldflags=['-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft'],
            verbose=False)
    return fwi
