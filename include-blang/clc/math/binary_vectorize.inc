#define __CLC_BINARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE, ARG2_TYPE) \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x, ARG2_TYPE##2 y) { \
    return (RET_TYPE##2)(FUNCTION(x.x, y.x), FUNCTION(x.y, y.y)); \
  } \
\
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x, ARG2_TYPE##3 y) { \
    return (RET_TYPE##3)(FUNCTION(x.x, y.x), FUNCTION(x.y, y.y), \
                         FUNCTION(x.z, y.z)); \
  } \
\
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x, ARG2_TYPE##4 y) { \
    return (RET_TYPE##4)(FUNCTION(x.lo, y.lo), FUNCTION(x.hi, y.hi)); \
  } \
\
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x, ARG2_TYPE##8 y) { \
    return (RET_TYPE##8)(FUNCTION(x.lo, y.lo), FUNCTION(x.hi, y.hi)); \
  } \
\
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x, ARG2_TYPE##16 y) { \
    return (RET_TYPE##16)(FUNCTION(x.lo, y.lo), FUNCTION(x.hi, y.hi)); \
  }
