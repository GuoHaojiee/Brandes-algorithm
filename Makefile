# Makefile - Brandes BC (MPI + CUDA)
# 平台: IBM Polus, ppc64le, CUDA 10.2, P100 (sm_60)
# 使用前请执行: module load mpi/openmpi3-ppc64le
#
# 若 CUDA 安装路径不同，可覆盖: make CUDA_HOME=/path/to/cuda

CUDA_HOME ?= /usr/local/cuda-10.2
NVCC       = $(CUDA_HOME)/bin/nvcc
MPICXX     = mpicxx

# nvcc 编译选项: sm_60 对应 P100，C++11，优化级别 O2
NVCC_FLAGS = -arch=sm_60 -O2 -std=c++11

# mpicxx 编译选项: 需要包含 CUDA 头文件路径
CXX_FLAGS  = -O2 -std=c++11 -I$(CUDA_HOME)/include

all: brandes

# 编译 CUDA 核函数目标文件
brandes_gpu.o: brandes_gpu.cu graph.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# 编译 MPI 主程序目标文件
main.o: main.cpp graph.h
	$(MPICXX) $(CXX_FLAGS) -c $< -o $@

# 链接: 用 mpicxx 链接两个目标文件，显式链接 CUDA 运行时库
brandes: main.o brandes_gpu.o
	$(MPICXX) $^ -L$(CUDA_HOME)/lib64 -lcudart -o $@

clean:
	rm -f *.o brandes

.PHONY: all clean
