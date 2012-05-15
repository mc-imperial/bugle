#include <bugle.h>

void foo(int x) {
  bugle_assert(x*2 == x<<1);
}
