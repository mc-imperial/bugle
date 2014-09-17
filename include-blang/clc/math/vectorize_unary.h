#define _CLC_VECTORIZE(FUNCTION, TYPE) \
  TYPE __uninterpreted_function_##FUNCTION##_##TYPE(TYPE x); \
  _CLC_INLINE _CLC_OVERLOAD TYPE FUNCTION(TYPE x) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE(x); \
  } \
\
  TYPE##2 __uninterpreted_function_##FUNCTION##_##TYPE##2(TYPE##2 x); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##2 FUNCTION(TYPE##2 x) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##2(x); \
  } \
\
  TYPE##3 __uninterpreted_function_##FUNCTION##_##TYPE##3(TYPE##3 x); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##3 FUNCTION(TYPE##3 x) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##3(x); \
  } \
\
  TYPE##4 __uninterpreted_function_##FUNCTION##_##TYPE##4(TYPE##4 x); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##4 FUNCTION(TYPE##4 x) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##4(x); \
  } \
\
  TYPE##8 __uninterpreted_function_##FUNCTION##_##TYPE##8(TYPE##8 x); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##8 FUNCTION(TYPE##8 x) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##8(x); \
  } \
\
  TYPE##16 __uninterpreted_function_##FUNCTION##_##TYPE##16(TYPE##16 x); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##16 FUNCTION(TYPE##16 x) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##16(x); \
  }
