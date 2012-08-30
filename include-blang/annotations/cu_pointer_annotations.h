#ifndef CU_POINTER_ANNOTATIONS_H
#define CU_POINTER_ANNOTATIONS_H

#ifndef __CUDA_ARCH__
#error For CUDA pointer annotations header to be included, __CUDA_ARCH__ must be defined
#endif

#define _POINTER_QUERY(NAME, TYPE) \
    _DEVICE_QUALIFIER TYPE __##NAME(const void* p);
    
#endif
