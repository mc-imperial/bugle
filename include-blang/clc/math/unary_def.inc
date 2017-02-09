#define __CLC_DEFINE_FUNC(RET_TYPE, FUNC, ARG_TYPE) \
  RET_TYPE __uninterpreted_function_##FUNC##_##ARG_TYPE(ARG_TYPE x); \
  \
  _CLC_INLINE _CLC_OVERLOAD RET_TYPE FUNC(ARG_TYPE x) { \
    return __uninterpreted_function_##FUNC##_##ARG_TYPE(x); \
  }

#define __CLC_UNARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG_TYPE) \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG_TYPE##2 x) { \
    return (RET_TYPE##2)(FUNCTION(x.x), FUNCTION(x.y)); \
  } \
\
  DECLSPEC RET_TYPE##3 FUNCTION(ARG_TYPE##3 x) { \
    return (RET_TYPE##3)(FUNCTION(x.x), FUNCTION(x.y), FUNCTION(x.z)); \
  } \
\
  DECLSPEC RET_TYPE##4 FUNCTION(ARG_TYPE##4 x) { \
    return (RET_TYPE##4)(FUNCTION(x.lo), FUNCTION(x.hi)); \
  } \
\
  DECLSPEC RET_TYPE##8 FUNCTION(ARG_TYPE##8 x) { \
    return (RET_TYPE##8)(FUNCTION(x.lo), FUNCTION(x.hi)); \
  } \
\
  DECLSPEC RET_TYPE##16 FUNCTION(ARG_TYPE##16 x) { \
    return (RET_TYPE##16)(FUNCTION(x.lo), FUNCTION(x.hi)); \
  }

#define __CLC_DECLARE_FUNC(RET_TYPE, FUNC, ARG_TYPE) \
  __CLC_DEFINE_FUNC(RET_TYPE, FUNC, ARG_TYPE) \
  __CLC_UNARY_VECTORIZE(_CLC_INLINE _CLC_OVERLOAD, RET_TYPE, FUNC, ARG_TYPE)

#ifndef __CLC_RET_TYPE
__CLC_DECLARE_FUNC(float, __CLC_FUNC_NAME, float)
#else
__CLC_DECLARE_FUNC(__CLC_RET_TYPE, __CLC_FUNC_NAME, float)
#endif

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef __CLC_RET_TYPE
__CLC_DECLARE_FUNC(double, __CLC_FUNC_NAME, double)
#else
__CLC_DECLARE_FUNC(__CLC_RET_TYPE, __CLC_FUNC_NAME, double)
#endif

#endif
#endif

#undef __CLC_DEFINE_FUNC
#undef __CLC_UNARY_VECTORIZE
#undef __CLC_DECLARE_FUNC
#undef __CLC_FUNC_NAME
#undef __CLC_RET_TYPE
