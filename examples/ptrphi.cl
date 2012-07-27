#define __1D_WORK_GROUP
#define __1D_GRID
#include <opencl.h>

void bar(__global int *ip) {
  ip[1] = 1;
}

__kernel void foo(bool p, __global int *ip) {
  if (p)
    ip++;
  while (ip[0])
    ip++;
  bar(ip);
}
