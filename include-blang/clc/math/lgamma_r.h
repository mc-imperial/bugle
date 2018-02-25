#define __CLC_DEFINE_UNINTERPRETED_LGAMMA_R(TYPE) \
  TYPE __uninterpreted_function_lgamma_##TYPE(TYPE x); \
  int __uninterpreted_function_lgamma_##TYPE##_sign(TYPE x);

#define __CLC_DEFINE_LGAMMA_R(DECLSPEC, TYPE, ADDRSPACE) \
  DECLSPEC TYPE __bugle_lgamma_r(TYPE x, ADDRSPACE int *signp) { \
    *signp = __uninterpreted_function_lgamma_##TYPE##_sign(x); \
    return __uninterpreted_function_lgamma_##TYPE(x); \
  } \
\
  DECLSPEC TYPE##2 __bugle_lgamma_r(TYPE##2 x, ADDRSPACE int2 *signp) { \
    *signp = (int2)(__uninterpreted_function_lgamma_##TYPE##_sign(x.x), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.y)); \
    return (TYPE##2)(__uninterpreted_function_lgamma_##TYPE(x.x), \
                     __uninterpreted_function_lgamma_##TYPE(x.y)); \
  } \
\
  DECLSPEC TYPE##3 __bugle_lgamma_r(TYPE##3 x, ADDRSPACE int3 *signp) { \
    *signp = (int3)(__uninterpreted_function_lgamma_##TYPE##_sign(x.x), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.y), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.z)); \
    return (TYPE##3)(__uninterpreted_function_lgamma_##TYPE(x.x), \
                     __uninterpreted_function_lgamma_##TYPE(x.y), \
                     __uninterpreted_function_lgamma_##TYPE(x.z)); \
  } \
\
  DECLSPEC TYPE##4 __bugle_lgamma_r(TYPE##4 x, ADDRSPACE int4 *signp) { \
    *signp = (int4)(__uninterpreted_function_lgamma_##TYPE##_sign(x.x), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.y), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.z), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.w)); \
    return (TYPE##4)(__uninterpreted_function_lgamma_##TYPE(x.x), \
                     __uninterpreted_function_lgamma_##TYPE(x.y), \
                     __uninterpreted_function_lgamma_##TYPE(x.z), \
                     __uninterpreted_function_lgamma_##TYPE(x.w)); \
  } \
\
  DECLSPEC TYPE##8 __bugle_lgamma_r(TYPE##8 x, ADDRSPACE int8 *signp) { \
    *signp = (int8)(__uninterpreted_function_lgamma_##TYPE##_sign(x.s0), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.s1), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.s2), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.s3), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.s4), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.s5), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.s6), \
                    __uninterpreted_function_lgamma_##TYPE##_sign(x.s7)); \
    return (TYPE##8)(__uninterpreted_function_lgamma_##TYPE(x.s0), \
                     __uninterpreted_function_lgamma_##TYPE(x.s1), \
                     __uninterpreted_function_lgamma_##TYPE(x.s2), \
                     __uninterpreted_function_lgamma_##TYPE(x.s3), \
                     __uninterpreted_function_lgamma_##TYPE(x.s4), \
                     __uninterpreted_function_lgamma_##TYPE(x.s5), \
                     __uninterpreted_function_lgamma_##TYPE(x.s6), \
                     __uninterpreted_function_lgamma_##TYPE(x.s7)); \
  } \
\
  DECLSPEC TYPE##16 __bugle_lgamma_r(TYPE##16 x, ADDRSPACE int16 *signp) { \
    *signp = (int16)(__uninterpreted_function_lgamma_##TYPE##_sign(x.s0), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s1), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s2), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s3), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s4), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s5), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s6), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s7), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s8), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.s9), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.sa), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.sb), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.sc), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.sd), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.se), \
                     __uninterpreted_function_lgamma_##TYPE##_sign(x.sf)); \
    return (TYPE##16)(__uninterpreted_function_lgamma_##TYPE(x.s0), \
                      __uninterpreted_function_lgamma_##TYPE(x.s1), \
                      __uninterpreted_function_lgamma_##TYPE(x.s2), \
                      __uninterpreted_function_lgamma_##TYPE(x.s3), \
                      __uninterpreted_function_lgamma_##TYPE(x.s4), \
                      __uninterpreted_function_lgamma_##TYPE(x.s5), \
                      __uninterpreted_function_lgamma_##TYPE(x.s6), \
                      __uninterpreted_function_lgamma_##TYPE(x.s7), \
                      __uninterpreted_function_lgamma_##TYPE(x.s8), \
                      __uninterpreted_function_lgamma_##TYPE(x.s9), \
                      __uninterpreted_function_lgamma_##TYPE(x.sa), \
                      __uninterpreted_function_lgamma_##TYPE(x.sb), \
                      __uninterpreted_function_lgamma_##TYPE(x.sc), \
                      __uninterpreted_function_lgamma_##TYPE(x.sd), \
                      __uninterpreted_function_lgamma_##TYPE(x.se), \
                      __uninterpreted_function_lgamma_##TYPE(x.sf)); \
  }

#define __CLC_DECLARE_LGAMMA_R(TYPE) \
  __CLC_DEFINE_UNINTERPRETED_LGAMMA_R(TYPE) \
  __CLC_DEFINE_LGAMMA_R(_CLC_INLINE _CLC_OVERLOAD, TYPE, global) \
  __CLC_DEFINE_LGAMMA_R(_CLC_INLINE _CLC_OVERLOAD, TYPE, local) \
  __CLC_DEFINE_LGAMMA_R(_CLC_INLINE _CLC_OVERLOAD, TYPE, private)

__CLC_DECLARE_LGAMMA_R(float)

#ifndef __FLOAT_ONLY
#ifdef cl_khr_fp64

__CLC_DECLARE_LGAMMA_R(double)

#endif
#endif

#undef __CLC_DEFINE_UNINTERPRETED_LGAMMA_R
#undef __CLC_DEFINE_LGAMMA_R
#undef __CLC_DECLARE_LGAMMA_R

#define lgamma_r __bugle_lgamma_r
