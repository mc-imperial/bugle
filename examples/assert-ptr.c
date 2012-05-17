#include <bugle.h>

void foo(_Bool p, unsigned i) {
  int x[4], y[4], *ptr = p ? x : y;
  ptr[0] = 0;
  ptr[1] = 1;
  ptr[2] = 2;
  ptr[3] = 3;
  bugle_assert(i >= 4 | ptr[i] == i);
}
