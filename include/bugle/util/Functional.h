#ifndef BUGLE_UTIL_FUNCTIONAL_H
#define BUGLE_UTIL_FUNCTIONAL_H

namespace bugle {

template <typename T, typename I, typename F>
T fold(T init, I begin, I end, F func) {
  T value = init;
  for (I i = begin; i != end; ++i)
    value = func(value, *i);
  return value;
}
}

#endif
