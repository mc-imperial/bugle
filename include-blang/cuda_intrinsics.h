#ifndef CUDA_INTRINSICS_H
#define CUDA_INTRINSICS_H

#if __CUDA_ARCH__ >= 300

__device__ int __shfl(int var, int srcLane, int width=warpSize);
__device__ int __shfl_up(int var, unsigned int delta, int width=warpSize);
__device__ int __shfl_down(int var, unsigned int delta, int width=warpSize);
__device__ int __shfl_xor(int var, int laneMask, int width=warpSize);
__device__ float __shfl(float var, int srcLane, int width=warpSize);
__device__ float __shfl_up(float var, unsigned int delta, int width=warpSize);
__device__ float __shfl_down(float var, unsigned int delta, int width=warpSize);
__device__ float __shfl_xor(float var, int laneMask, int width=warpSize);

#endif

#endif
