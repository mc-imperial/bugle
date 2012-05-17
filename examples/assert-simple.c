#include <bugle.h>

void foo(unsigned i) {
  int x[4];
  x[0] = 0;
  x[1] = 1;
  x[2] = 2;
  x[3] = 3;
  bugle_assert(i >= 4 | x[i] == i);
}
