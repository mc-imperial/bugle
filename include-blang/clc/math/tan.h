#define tan __bugle_tan

#include <clc/math/vectorize_unary.h>

_CLC_VECTORIZE(__bugle_tan, float)

#ifdef cl_khr_fp64
_CLC_VECTORIZE(__bugle_tan, double)
#endif

#undef _CLC_VECTORIZE
