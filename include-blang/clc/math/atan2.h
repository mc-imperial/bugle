#define atan2 __bugle_atan2

#include <clc/math/vectorize_binary.h>

_CLC_VECTORIZE(__bugle_atan2, float)

#ifdef cl_khr_fp64
_CLC_VECTORIZE(__bugle_atan2, double)
#endif

#undef _CLC_VECTORIZE
