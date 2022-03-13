#ifndef CUDA_H
#define CUDA_H

#include <stddef.h>

#pragma GCC diagnostic error "-Wimplicit-function-declaration"
#pragma GCC diagnostic ignored "-Wc++11-long-long"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#ifndef __CUDA_ARCH__
#error __CUDA_ARCH__ must be defined
#endif

#ifdef __OPENCL_VERSION__
#error Cannot include both opencl.h and cuda.h
#endif

#define __CUDACC__

#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __inline__ __attribute__((always_inline))
#define __forceinline__ __attribute__((always_inline))

#ifdef __cplusplus
extern "C" {
#endif

struct _3DimensionalVector {
  unsigned x, y, z;
} threadIdx, blockIdx, blockDim, gridDim;

#if __CUDA_ARCH__ >= 300
int warpSize;
#endif

#define __syncthreads() \
  bugle_barrier(true, true)

__device__ void __threadfence_block();
__device__ void __threadfence();
__device__ void __threadfence_system();

#ifdef __cplusplus
}
#endif

/* Use an empty definition for alignment. Alternatively we could use:

     #define __align__(n) __attribute__((aligned(n)))

   but this causes bugle to default to byte-size operations even for larger
   data types.
*/
#define __align__(n)

#define __launch_bounds__(x, y)

#include <bugle.h>
#include <annotations/annotations.h>
#include <cuda_math_constants.h>
#include <cuda_math_functions.h>
#include <cuda_vectors.h>
#include <cuda_textures.h>
#include <cuda_atomics.h>
#include <cuda_curand.h>
#include <cuda_intrinsics.h>
#include <cuda_cooperative_groups.h>

/* Thread block dimensions */

// Must define a dimension

#ifndef __1D_THREAD_BLOCK
#ifndef __2D_THREAD_BLOCK
#ifndef __3D_THREAD_BLOCK

#error You must specify the dimension of a work group by defining one of __1D_THREAD_BLOCK, __2D_THREAD_BLOCK or __3D_THREAD_BLOCK

#endif
#endif
#endif

// Must define only one dimension

#ifdef __1D_THREAD_BLOCK
#ifdef __2D_THREAD_BLOCK
#error Cannot define __1D_THREAD_BLOCK and __2D_THREAD_BLOCK
#endif
#ifdef __3D_THREAD_BLOCK
#error Cannot define __1D_THREAD_BLOCK and __3D_THREAD_BLOCK
#endif
#endif

#ifdef __2D_THREAD_BLOCK
#ifdef __1D_THREAD_BLOCK
#error Cannot define __2D_THREAD_BLOCK and __1D_THREAD_BLOCK
#endif
#ifdef __3D_THREAD_BLOCK
#error Cannot define __2D_THREAD_BLOCK and __3D_THREAD_BLOCK
#endif
#endif

#ifdef __3D_THREAD_BLOCK
#ifdef __1D_THREAD_BLOCK
#error Cannot define __3D_THREAD_BLOCK and __1D_THREAD_BLOCK
#endif
#ifdef __2D_THREAD_BLOCK
#error Cannot define __3D_THREAD_BLOCK and __2D_THREAD_BLOCK
#endif
#endif

// Generate axioms for different work group sizes

#ifdef __1D_THREAD_BLOCK
__axiom(blockDim.y == 1)
__axiom(blockDim.z == 1)
#endif

#ifdef __2D_THREAD_BLOCK
__axiom(blockDim.z == 1)
#endif


/* Thread block grid dimensions */

// Must define a dimension

#ifndef __1D_GRID
#ifndef __2D_GRID
#ifndef __3D_GRID

#error You must specify the dimension of the grid of thread blocks by defining one of __1D_GRID, __2D_GRID or __3D_GRID

#endif
#endif
#endif

// Must define only one dimension

#ifdef __1D_GRID
#ifdef __2D_GRID
#error Cannot define __1D_GRID and __2D_GRID
#endif
#ifdef __3D_GRID
#error Cannot define __1D_GRID and __3D_GRID
#endif
#endif

#ifdef __2D_GRID
#ifdef __1D_GRID
#error Cannot define __2D_GRID and __1D_GRID
#endif
#ifdef __3D_GRID
#error Cannot define __2D_GRID and __3D_GRID
#endif
#endif

#ifdef __3D_GRID
#ifdef __1D_GRID
#error Cannot define __3D_GRID and __1D_GRID
#endif
#ifdef __2D_GRID
#error Cannot define __3D_GRID and __2D_GRID
#endif
#endif

// Generate axioms for different grid sizes

#ifdef __1D_GRID
__axiom(gridDim.y == 1)
__axiom(gridDim.z == 1)
#endif

#ifdef __2D_GRID
__axiom(gridDim.z == 1)
#endif

// Generate axioms for input values

#if defined(__BLOCK_DIM_0) && defined(__BLOCK_DIM_0_FREE)
#error Cannot define __BLOCK_DIM_0 and __BLOCK_DIM_0_FREE
#elif defined(__BLOCK_DIM_0)
__axiom(blockDim.x == __BLOCK_DIM_0)
#elif defined(__BLOCK_DIM_0_FREE)
__axiom(blockDim.x > 0)
#endif

#if defined(__BLOCK_DIM_1) && defined(__BLOCK_DIM_1_FREE)
#error Cannot define __BLOCK_DIM_1 and __BLOCK_DIM_1_FREE
#elif defined(__BLOCK_DIM_1)
__axiom(blockDim.y == __BLOCK_DIM_1)
#elif defined(__BLOCK_DIM_1_FREE)
__axiom(blockDim.y > 0)
#endif

#if defined(__BLOCK_DIM_2) && defined(__BLOCK_DIM_2_FREE)
#error Cannot define __BLOCK_DIM_2 and __BLOCK_DIM_2_FREE
#elif defined(__BLOCK_DIM_2)
__axiom(blockDim.z == __BLOCK_DIM_2)
#elif defined(__BLOCK_DIM_2_FREE)
__axiom(blockDim.z > 0)
#endif

#if defined(__GRID_DIM_0) && defined(__GRID_DIM_0_FREE)
#error Cannot define __GRID_DIM_0 and __GRID_DIM_0_FREE
#elif defined(__GRID_DIM_0)
__axiom(gridDim.x == __GRID_DIM_0)
#elif defined(__GRID_DIM_0_FREE)
__axiom(gridDim.x > 0)
#endif

#if defined(__GRID_DIM_1) && defined(__GRID_DIM_1_FREE)
#error Cannot define __GRID_DIM_1 and __GRID_DIM_1_FREE
#elif defined(__GRID_DIM_1)
__axiom(gridDim.y == __GRID_DIM_1)
#elif defined(__GRID_DIM_1_FREE)
__axiom(gridDim.y > 0)
#endif

#if defined(__GRID_DIM_2) && defined(__GRID_DIM_2_FREE)
#error Cannot define __GRID_DIM_2 and __GRID_DIM_2_FREE
#elif defined(__GRID_DIM_2)
__axiom(gridDim.z == __GRID_DIM_2)
#elif defined(__GRID_DIM_2_FREE)
__axiom(gridDim.z > 0)
#endif

/* Warp size */

// Must define a warp size

#ifndef __WARP_SIZE
#error You must specify the warp size by defining __WARP_SIZE
#endif

#if __CUDA_ARCH__ >= 300
__axiom(warpSize == __WARP_SIZE)
#endif

#pragma GCC diagnostic pop

#endif
