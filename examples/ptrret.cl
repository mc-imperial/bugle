#define __1D_WORK_GROUP
#define __1D_GRID
#include <opencl.h>

__global int *bar(bool p, __global int *ip) {
  return p ? ip+1 : ip+3;
}

__kernel void foo(__global int *ip, __global int *jp) {
  bar(true, ip)[1] = 42;;
}
