#define atan __bugle_atan

#include <clc/math/vectorize_unary.h>

_CLC_VECTORIZE(__bugle_atan, float)

#ifdef cl_khr_fp64
_CLC_VECTORIZE(__bugle_atan, double)
#endif

#undef _CLC_VECTORIZE
