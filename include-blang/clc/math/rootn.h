#include <clc/math/binary_vectorize.inc>

#define __CLC_DEFINE_ROOTN(GENTYPE, INTTYPE) \
  GENTYPE \
  __uninterpreted_function___bugle_rootn_##GENTYPE(GENTYPE x, INTTYPE y); \
  \
  _CLC_INLINE _CLC_OVERLOAD GENTYPE __bugle_rootn(GENTYPE x, INTTYPE y) { \
    return __uninterpreted_function___bugle_rootn_##GENTYPE(x, y); \
  }

#define __CLC_DEFINE_VECTOR_ROOTN(GENTYPE, INTTYPE) \
  __CLC_BINARY_VECTORIZE( \
    _CLC_INLINE _CLC_OVERLOAD, GENTYPE, __bugle_rootn, GENTYPE, INTTYPE)

__CLC_DEFINE_ROOTN(float, int)
__CLC_DEFINE_VECTOR_ROOTN(float, int)

#ifdef cl_khr_fp64

__CLC_DEFINE_ROOTN(double, int)
__CLC_DEFINE_VECTOR_ROOTN(double, int)

#endif

#undef __CLC_DEFINE_ROOTN
#undef __CLC_DEFINE_VECTOR_ROOTN
#undef __CLC_BINARY_VECTORIZE

#define rootn __bugle_rootn

