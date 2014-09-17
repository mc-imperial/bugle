#define asin __bugle_asin

#include <clc/math/vectorize_unary.h>

_CLC_VECTORIZE(__bugle_asin, float)

#ifdef cl_khr_fp64
_CLC_VECTORIZE(__bugle_asin, double)
#endif

#undef _CLC_VECTORIZE
