#ifndef CL_POINTER_ANNOTATIONS_H
#define CL_POINTER_ANNOTATIONS_H

#ifndef __OPENCL__
#error For OpenCL pointer annotations header to be included, __OPENCL__ must be defined
#endif

#define __POINTER_QUERY_OVERLOAD(NAME, MEMORY_SPACE, TYPE) \
    TYPE \
    __##NAME##_##MEMORY_SPACE \
    (const __##MEMORY_SPACE void* p); \
    __attribute__((overloadable)) \
    static __attribute__((always_inline)) TYPE \
    __##NAME(const __##MEMORY_SPACE void* p) { \
      return __##NAME##_##MEMORY_SPACE(p); \
    }

#define __POINTER_QUERY(NAME, TYPE) \
    __POINTER_QUERY_OVERLOAD(NAME, local, TYPE) \
    __POINTER_QUERY_OVERLOAD(NAME, global, TYPE) \
    
#endif
