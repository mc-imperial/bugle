#include <clc/math/binary_vectorize.inc>

#define __CLC_POWN_INTRINSIC "llvm.powi"

#define __CLC_DEFINE_VECTOR_POWN(GENTYPE, INTTYPE) \
  __CLC_BINARY_VECTORIZE( \
    _CLC_INLINE _CLC_OVERLOAD, GENTYPE, __bugle_pown, GENTYPE, INTTYPE)

_CLC_OVERLOAD
float __bugle_pown(float x, int y) __asm(__CLC_POWN_INTRINSIC ".f32");

__CLC_DEFINE_VECTOR_POWN(float, int)

#ifdef cl_khr_fp64

_CLC_OVERLOAD
double __bugle_pown(double x, int y) __asm(__CLC_POWN_INTRINSIC ".f64");

__CLC_DEFINE_VECTOR_POWN(double, int)

#endif

#undef __CLC_POWN_INTRINSIC
#undef __CLC_DEFINE_VECTOR_POWN
#undef __CLC_BINARY_VECTORIZE

#define pown __bugle_pown
