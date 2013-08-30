#ifndef CUDA_H
#define CUDA_H

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

/* Use an empty definition for alignment. Alternatively we could use:

     #define __align__(n) __attribute__((aligned(n)))

   but this causes bugle to default to byte-size operations even for larger
   data types.
*/
#define __align__(n)

#define NULL 0
typedef unsigned int size_t;

#include <bugle.h>
#include <annotations/annotations.h>
#include <cuda_math_constants.h>
#include <cuda_math_functions.h>
#include <cuda_vectors.h>
#include <cuda_textures.h>
#include <cuda_atomics.h>


#ifdef __cplusplus
extern "C" {
#endif

struct _3DimensionalVector {
  unsigned x, y, z;
} threadIdx, blockIdx, blockDim, gridDim;

#define __syncthreads() \
  bugle_barrier(true, true)

#ifdef __cplusplus
}
#endif

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
__axiom(blockDim.y == 1);
__axiom(blockDim.z == 1);
#endif

#ifdef __2D_THREAD_BLOCK
__axiom(blockDim.z == 1);
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
__axiom(gridDim.y == 1);
__axiom(gridDim.z == 1);
#endif

#ifdef __2D_GRID
__axiom(gridDim.z == 1);
#endif

#ifdef __BLOCK_DIM_0
__axiom(blockDim.x == __BLOCK_DIM_0)
#endif

#ifdef __BLOCK_DIM_1
__axiom(blockDim.y == __BLOCK_DIM_1)
#endif

#ifdef __BLOCK_DIM_2
__axiom(blockDim.z == __BLOCK_DIM_2)
#endif

#ifdef __GRID_DIM_0
__axiom(gridDim.x == __GRID_DIM_0)
#endif

#ifdef __GRID_DIM_1
__axiom(gridDim.y == __GRID_DIM_1)
#endif

#ifdef __GRID_DIM_2
__axiom(gridDim.z == __GRID_DIM_2)
#endif

#endif
