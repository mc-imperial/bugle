#define __1D_WORK_GROUP
#define __1D_GRID
#include <opencl.h>

void bar(__global int *ip) {
  ip[1] = 1;
}

void baz(__global int *jp) {
  ((__global char *)jp)[1] = 1;
}

__kernel void foo(__global int *ip, __global int *jp) {
  bar(ip);
  bar(ip+1);
  baz(jp);
  baz(jp+1);
}
