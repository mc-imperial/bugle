#include <bugle.h>

void foo() {
  unsigned x = 1;
  while (bugle_assert(!(x & (x - 1)) & (x <= 64)), x < 64)
    x *= 2;
  bugle_assert(x == 64);
}
