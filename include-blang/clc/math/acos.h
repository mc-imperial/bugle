#define acos __bugle_acos

#include <clc/math/vectorize_unary.h>

_CLC_VECTORIZE(__bugle_acos, float)

#ifdef cl_khr_fp64
_CLC_VECTORIZE(__bugle_acos, double)
#endif

#undef _CLC_VECTORIZE
