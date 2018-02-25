#include <clc/math/binary_vectorize.inc>

#define __CLC_DEFINE_FUNC(FUNC, TYPE) \
  TYPE __uninterpreted_function_##FUNC##_##TYPE(TYPE x, TYPE y); \
  \
  _CLC_INLINE _CLC_OVERLOAD TYPE FUNC(TYPE x, TYPE y) { \
    return __uninterpreted_function_##FUNC##_##TYPE(x, y); \
  }

#define __CLC_DECLARE_FUNC(FUNC, TYPE) \
  __CLC_DEFINE_FUNC(FUNC, TYPE) \
  __CLC_BINARY_VECTORIZE(_CLC_INLINE _CLC_OVERLOAD, TYPE, FUNC, TYPE, TYPE)

__CLC_DECLARE_FUNC(__CLC_FUNC_NAME, float)

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64

__CLC_DECLARE_FUNC(__CLC_FUNC_NAME, double)

#endif
#endif

#undef __CLC_DEFINE_FUNC
#undef __CLC_DECLARE_FUNC
#undef __CLC_BINARY_VECTORIZE
#undef __CLC_FUNC_NAME
