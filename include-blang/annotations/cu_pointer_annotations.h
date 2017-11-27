#ifndef CU_POINTER_ANNOTATIONS_H
#define CU_POINTER_ANNOTATIONS_H

#ifndef __CUDA_ARCH__
#error For CUDA pointer annotations header to be included, __CUDA_ARCH__ must be defined
#endif

#define _FUNCTION_FROM_POINTER_TO_TYPE(NAME, TYPE) \
    _DEVICE_QUALIFIER TYPE __##NAME(const volatile void* p);

#define _FUNCTION_FROM_POINTER_TO_VOID(NAME) \
    _DEVICE_QUALIFIER void __##NAME(const volatile void* p);

#endif
