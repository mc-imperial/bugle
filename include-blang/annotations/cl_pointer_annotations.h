#ifndef CL_POINTER_ANNOTATIONS_H
#define CL_POINTER_ANNOTATIONS_H

#ifndef __OPENCL_VERSION__
#error For OpenCL pointer annotations header to be included, __OPENCL_VERSION__ must be defined
#endif

#define _POINTER_QUERY_OVERLOAD(NAME, MEMORY_SPACE, TYPE) \
    TYPE \
    __##NAME##_##MEMORY_SPACE \
    (const __##MEMORY_SPACE void* p); \
    _CLC_OVERLOAD _CLC_INLINE TYPE \
    __##NAME(const __##MEMORY_SPACE void* p) { \
      return __##NAME##_##MEMORY_SPACE(p); \
    }

#define _POINTER_QUERY(NAME, TYPE) \
    _POINTER_QUERY_OVERLOAD(NAME, local, TYPE) \
    _POINTER_QUERY_OVERLOAD(NAME, global, TYPE) \
    
#endif
