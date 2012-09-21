#ifndef OPENCL_H
#define OPENCL_H

#ifndef __OPENCL_VERSION__
#error __OPENCL_VERSION__ must be defined
#endif

#ifdef __CUDA_ARCH__
#error Cannot include both opencl.h and cuda.h
#endif

#include <bugle.h>
#include <clc/clc.h>

#pragma OPENCL EXTENSION cl_khr_fp64: enable

/* Images */
typedef __global void *image2d_t;
typedef __global void *image3d_t;

#define CLK_NORMALIZED_COORDS_TRUE  0
#define CLK_NORMALIZED_COORDS_FALSE 1
#define CLK_ADDRESS_MIRRORED_REPEAT 0
#define CLK_ADDRESS_REPEAT          2
#define CLK_ADDRESS_CLAMP_TO_EDGE   4
#define CLK_ADDRESS_CLAMP           6
#define CLK_ADDRESS_NONE            8
#define CLK_FILTER_NEAREST          0
#define CLK_FILTER_LINEAR           16

typedef unsigned int sampler_t;

#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers: enable

_CLC_INLINE _CLC_OVERLOAD float4 read_imagef(image2d_t image, sampler_t sampler, uint2 coord) {
  __global float4 *img = image;
  return img[coord.y*(2<<16) + coord.x];
}
_CLC_INLINE _CLC_OVERLOAD float4 read_imagef(image2d_t image, sampler_t sampler, int2 coord) {
  __global float4 *img = image;
  return img[coord.y*(2<<16) + coord.x];
}
_CLC_INLINE int4 read_imagei(image2d_t image, sampler_t sampler, uint2 coord) {
  __global int4 *img = image;
  return img[coord.y*(2<<16) + coord.x];
}
_CLC_INLINE uint4 read_imageui(image2d_t image, sampler_t sampler, uint2 coord) {
  __global uint4 *img = image;
  return img[coord.y*(2<<16) + coord.x];
}

_CLC_INLINE void write_imagef(image2d_t image, uint2 coord, float4 color) {
  __global float4 *img = image;
  img[coord.y*(2<<16) + coord.x] = color;
}
_CLC_INLINE void write_imagei(image2d_t image, uint2 coord, int4 color) {
  __global int4 *img = image;
  img[coord.y*(2<<16) + coord.x] = color;
}
_CLC_INLINE void write_imageui(image2d_t image, uint2 coord, uint4 color) {
  __global uint4 *img = image;
  img[coord.y*(2<<16) + coord.x] = color;
}

#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers: disable

int get_image_height (image2d_t image);
int get_image_width (image2d_t image);

#include "annotations/annotations.h"

// Must define a dimension

#ifndef __1D_WORK_GROUP
#ifndef __2D_WORK_GROUP
#ifndef __3D_WORK_GROUP

#error You must specify the dimension of a work group by defining one of __1D_WORK_GROUP, __2D_WORK_GROUP or __3D_WORK_GROUP

#endif
#endif
#endif

// Must define only one dimension

#ifdef __1D_WORK_GROUP
#ifdef __2D_WORK_GROUP
#error Cannot define __1D_WORK_GROUP and __2D_WORK_GROUP
#endif
#ifdef __3D_WORK_GROUP
#error Cannot define __1D_WORK_GROUP and __3D_WORK_GROUP
#endif
#endif

#ifdef __2D_WORK_GROUP
#ifdef __1D_WORK_GROUP
#error Cannot define __2D_WORK_GROUP and __1D_WORK_GROUP
#endif
#ifdef __3D_WORK_GROUP
#error Cannot define __2D_WORK_GROUP and __3D_WORK_GROUP
#endif
#endif

#ifdef __3D_WORK_GROUP
#ifdef __1D_WORK_GROUP
#error Cannot define __3D_WORK_GROUP and __1D_WORK_GROUP
#endif
#ifdef __2D_WORK_GROUP
#error Cannot define __3D_WORK_GROUP and __2D_WORK_GROUP
#endif
#endif

// Generate axioms for different work group sizes

#ifdef __1D_WORK_GROUP
__axiom(get_local_size(1) == 1);
__axiom(get_local_size(2) == 1);
#endif

#ifdef __2D_WORK_GROUP
__axiom(get_local_size(2) == 1);
#endif


/* Work group grid dimensions */

// Must define a dimension

#ifndef __1D_GRID
#ifndef __2D_GRID
#ifndef __3D_GRID

#error You must specify the dimension of the grid of work groups by defining one of __1D_GRID, __2D_GRID or __3D_GRID

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
__axiom(get_num_groups(1) == 1);
__axiom(get_num_groups(2) == 1);
#endif

#ifdef __2D_GRID
__axiom(get_num_groups(2) == 1);
#endif



#endif
