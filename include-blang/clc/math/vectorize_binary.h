#define _CLC_VECTORIZE(FUNCTION, TYPE) \
  TYPE __uninterpreted_function_##FUNCTION##_##TYPE(TYPE x, TYPE y); \
  _CLC_INLINE _CLC_OVERLOAD TYPE FUNCTION(TYPE x, TYPE y) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE(x, y); \
  } \
\
  TYPE##2 __uninterpreted_function_##FUNCTION##_##TYPE##2(TYPE##2 x, \
                                                          TYPE##2 y); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##2 FUNCTION(TYPE##2 x, TYPE##2 y) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##2(x, y); \
  } \
\
  TYPE##3 __uninterpreted_function_##FUNCTION##_##TYPE##3(TYPE##3 x, \
                                                          TYPE##3 y); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##3 FUNCTION(TYPE##3 x, TYPE##3 y) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##3(x, y); \
  } \
\
  TYPE##4 __uninterpreted_function_##FUNCTION##_##TYPE##4(TYPE##4 x, \
                                                          TYPE##4 y); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##4 FUNCTION(TYPE##4 x, TYPE##4 y) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##4(x, y); \
  } \
\
  TYPE##8 __uninterpreted_function_##FUNCTION##_##TYPE##8(TYPE##8 x, \
                                                          TYPE##8 y); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##8 FUNCTION(TYPE##8 x, TYPE##8 y) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##8(x, y); \
  } \
\
  TYPE##16 __uninterpreted_function_##FUNCTION##_##TYPE##16(TYPE##16 x, \
                                                            TYPE##16 y); \
  _CLC_INLINE _CLC_OVERLOAD TYPE##16 FUNCTION(TYPE##16 x, TYPE##16 y) { \
    return __uninterpreted_function_##FUNCTION##_##TYPE##16(x, y); \
  }
