#ifndef CUDA_INTRINSICS_H
#define CUDA_INTRINSICS_H

extern "C" __device__ int printf(const char *format, ...);

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

#if __CUDA_ARCH__ >= 350

static __device__ __inline__ float __ldg(const float *ptr) { return *ptr; }

#endif

#endif
