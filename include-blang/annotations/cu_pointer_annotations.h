#ifndef CU_POINTER_ANNOTATIONS_H
#define CU_POINTER_ANNOTATIONS_H

#ifndef __CUDA__
#error For CUDA pointer annotations header to be included, __CUDA__ must be defined
#endif

#define __POINTER_QUERY(NAME, TYPE) \
    __DEVICE_QUALIFIER__ TYPE __##NAME(const void* p);
    
#endif
