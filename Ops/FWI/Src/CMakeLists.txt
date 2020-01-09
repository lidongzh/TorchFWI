cmake_minimum_required(VERSION 3.5)

# change to your own cuda compiler
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_COMPILER nvcc)
SET(CMAKE_CXX_FLAGS "-std=gnu++11 -O3")
project(CUFD)
enable_language(CUDA)
include_directories(rapidjson)
include_directories(/usr/local/cuda/include)


# add_executable(CUFDMAIN Parameter.cpp main.cu add_source.cu el_stress.cu el_velocity.cu
# ac_pressure.cu ac_velocity.cu Model.cu Cpml.cu utilities.cu 
# Src_Rec.cu recording.cu Boundary.cu ac_pressure_adj.cu ac_velocity_adj.cu 
# res_injection.cu image_vel.cu image_vel_time.cu ac_adj_push.cu)
# target_link_libraries(CUFDMAIN nvrtc cuda cudart cufft)

add_library(CUFD SHARED Parameter.cpp libCUFD.cu el_stress.cu el_velocity.cu
ac_pressure.cu ac_velocity.cu Model.cu Cpml.cu utilities.cu 
Src_Rec.cu Boundary.cu ac_pressure_adj.cu ac_velocity_adj.cu 
image_vel.cu image_vel_time.cu ac_adj_push.cu el_stress_adj.cu 
el_velocity_adj.cu)
target_link_libraries(CUFD nvrtc cuda cudart cufft)
set_property(TARGET CUFD PROPERTY POSITION_INDEPENDENT_CODE ON)
