#define __CLC_DEFINE_FUNC(FUNC, TYPE) \
  TYPE __uninterpreted_function_##FUNC##_##TYPE(TYPE x, TYPE y); \
  \
  _CLC_INLINE _CLC_OVERLOAD TYPE FUNC(TYPE x, TYPE y) { \
    return __uninterpreted_function_##FUNC##_##TYPE(x, y); \
  }

#define __CLC_DECLARE_FUNC(FUNC, TYPE) \
  __CLC_DEFINE_FUNC(FUNC, TYPE)

__CLC_DECLARE_FUNC(__CLC_FUNC_NAME, __CLC_GENTYPE)

#undef __CLC_DEFINE_FUNC
#undef __CLC_DECLARE_FUNC
