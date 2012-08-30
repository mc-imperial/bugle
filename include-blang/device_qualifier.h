#ifdef __CUDA_ARCH__
#define _DEVICE_QUALIFIER __device__
#else
#define _DEVICE_QUALIFIER
#endif
