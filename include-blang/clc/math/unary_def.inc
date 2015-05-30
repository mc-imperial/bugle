#define __CLC_DEFINE_FUNC(FUNC, TYPE) \
  TYPE __uninterpreted_function_##FUNC##_##TYPE(TYPE x); \
  \
  _CLC_INLINE _CLC_OVERLOAD TYPE FUNC(TYPE x) { \
    return __uninterpreted_function_##FUNC##_##TYPE(x); \
  }

#define __CLC_UNARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE) \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x) { \
    return (RET_TYPE##2)(FUNCTION(x.x), FUNCTION(x.y)); \
  } \
\
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x) { \
    return (RET_TYPE##3)(FUNCTION(x.x), FUNCTION(x.y), FUNCTION(x.z)); \
  } \
\
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x) { \
    return (RET_TYPE##4)(FUNCTION(x.lo), FUNCTION(x.hi)); \
  } \
\
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x) { \
    return (RET_TYPE##8)(FUNCTION(x.lo), FUNCTION(x.hi)); \
  } \
\
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x) { \
    return (RET_TYPE##16)(FUNCTION(x.lo), FUNCTION(x.hi)); \
  }

#define __CLC_DECLARE_FUNC(FUNC, TYPE) \
  __CLC_DEFINE_FUNC(FUNC, TYPE) \
  __CLC_UNARY_VECTORIZE(_CLC_INLINE _CLC_OVERLOAD, TYPE, FUNC, TYPE)

__CLC_DECLARE_FUNC(__CLC_FUNC_NAME, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__CLC_DECLARE_FUNC(__CLC_FUNC_NAME, double)

#endif

#undef __CLC_DEFINE_FUNC
#undef __CLC_UNARY_VECTORIZE
#undef __CLC_DECLARE_FUNC
#undef __CLC_FUNC_NAME
