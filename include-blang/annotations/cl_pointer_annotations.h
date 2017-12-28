#ifndef CL_POINTER_ANNOTATIONS_H
#define CL_POINTER_ANNOTATIONS_H

#ifndef __OPENCL_VERSION__
#error For OpenCL pointer annotations header to be included, __OPENCL_VERSION__ must be defined
#endif

#define _FUNCTION_FROM_POINTER_TO_TYPE_OVERLOAD(NAME, MEMORY_SPACE, TYPE) \
    TYPE \
    __##NAME##_##MEMORY_SPACE \
    (const volatile __##MEMORY_SPACE void* p); \
    _CLC_OVERLOAD _CLC_INLINE TYPE \
    __##NAME(const volatile __##MEMORY_SPACE void* p) { \
      return __##NAME##_##MEMORY_SPACE(p); \
    }

#define _FUNCTION_FROM_POINTER_TO_TYPE(NAME, TYPE) \
    _FUNCTION_FROM_POINTER_TO_TYPE_OVERLOAD(NAME, local, TYPE) \
    _FUNCTION_FROM_POINTER_TO_TYPE_OVERLOAD(NAME, global, TYPE) \

#define _FUNCTION_FROM_POINTER_TO_VOID_OVERLOAD(NAME, MEMORY_SPACE) \
    void \
    __##NAME##_##MEMORY_SPACE \
    (const volatile __##MEMORY_SPACE void* p); \
    _CLC_OVERLOAD _CLC_INLINE void \
    __##NAME(const volatile __##MEMORY_SPACE void* p) { \
      __##NAME##_##MEMORY_SPACE(p); \
    }

#define _FUNCTION_FROM_POINTER_TO_VOID(NAME) \
    _FUNCTION_FROM_POINTER_TO_VOID_OVERLOAD(NAME, local) \
    _FUNCTION_FROM_POINTER_TO_VOID_OVERLOAD(NAME, global) \

#endif
