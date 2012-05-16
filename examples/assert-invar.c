#include <bugle.h>

void foo() {
  unsigned x = 0;
  while (bugle_assert(x <= 64), x < 64)
    x++;
  bugle_assert(x == 64);
}
