#ifndef CUDA_ATOMICS_H
#define CUDA_ATOMICS_H

#ifdef __CUDA_ARCH__

extern "C" {

#define ATOMIC_INT_DECL(OP) \
    _DEVICE_QUALIFIER int __atomic##OP##_int(volatile int * x, int y); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic##OP(volatile int * x, int y) { \
      return __atomic##OP##_int(x,y); \
    }

ATOMIC_INT_DECL(Add)
ATOMIC_INT_DECL(Sub)
ATOMIC_INT_DECL(Exch)
ATOMIC_INT_DECL(Min)
ATOMIC_INT_DECL(Max)
ATOMIC_INT_DECL(And)
ATOMIC_INT_DECL(Or)
ATOMIC_INT_DECL(Xor)

#undef ATOMIC_INT_DECL
#define ATOMIC_UNSIGNED_INT_DECL(OP) \
    _DEVICE_QUALIFIER unsigned int __atomic##OP##_unsigned_int(volatile unsigned int * x, unsigned int y); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned int atomic##OP(volatile unsigned int * x, unsigned int y) { \
      return __atomic##OP##_unsigned_int(x,y); \
    }

ATOMIC_UNSIGNED_INT_DECL(Add)
ATOMIC_UNSIGNED_INT_DECL(Sub)
ATOMIC_UNSIGNED_INT_DECL(Exch)
ATOMIC_UNSIGNED_INT_DECL(Min)
ATOMIC_UNSIGNED_INT_DECL(Max)
ATOMIC_UNSIGNED_INT_DECL(And)
ATOMIC_UNSIGNED_INT_DECL(Or)
ATOMIC_UNSIGNED_INT_DECL(Xor)
ATOMIC_UNSIGNED_INT_DECL(Inc)
ATOMIC_UNSIGNED_INT_DECL(Dec)

#undef ATOMIC_UNSIGNED_INT_DECL

#define ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(OP) \
    _DEVICE_QUALIFIER unsigned long long int __atomic##OP##_unsigned_long_long_int(volatile unsigned long long int * x, unsigned long long int y); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned long long int atomic##OP(volatile unsigned long long int * x, unsigned long long int y) { \
      return __atomic##OP##_unsigned_long_long_int(x,y); \
    }

ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(Add)
ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(Exch)
ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(Min)
ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(Max)
ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(And)
ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(Or)
ATOMIC_UNSIGNED_LONG_LONG_INT_DECL(Xor)

#undef ATOMIC_UNSIGNED_LONG_LONG_INT_DECL

#define ATOMIC_FLOAT_DECL(OP) \
    _DEVICE_QUALIFIER float __atomic##OP##_float(volatile float * x, float y); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) float atomic##OP(volatile float * x, float y) { \
      return __atomic##OP##_float(x,y); \
    }

ATOMIC_FLOAT_DECL(Add)
ATOMIC_FLOAT_DECL(Exch)

#undef ATOMIC_FLOAT_DECL

/* atomicCAS(x, y, z) */
_DEVICE_QUALIFIER int __atomicCAS_int(volatile int * x, int y, int z); \
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomicCAS(volatile int * x, int y, int z) {
  return __atomicCAS_int(x,y,z);
}
_DEVICE_QUALIFIER unsigned int __atomicCAS_unsigned_int(volatile unsigned int * x, unsigned int y, unsigned int z); \
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned int atomicCAS(volatile unsigned int * x, unsigned int y, unsigned int z) {
  return __atomicCAS_unsigned_int(x,y,z);
}
_DEVICE_QUALIFIER unsigned long long int __atomicCAS_unsigned_long_long_int(volatile unsigned long long int * x, unsigned long long int y, unsigned long long int z); \
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned long long int atomicCAS(volatile unsigned long long int * x, unsigned long long int y, unsigned long long int z) {
  return __atomicCAS_unsigned_long_long_int(x,y,z);
}


/*
 * Expected
 * ========
 *
 * atomicAdd: int, unsigned int, unsigned long long int, float
 * atomicSub: int, unsigned int
 * atomicExch: int, unsigned int, unsigned long long int, float
 * atomicMin: int, unsigned int, unsigned long long int
 * atomicMax: int, unsigned int, unsigned long long int
 *
 * atomicAnd: int, unsigned int, unsigned long long int
 * atomicOr: int, unsigned int, unsigned long long int
 * atomicXor: int, unsigned int, unsigned long long int
 *
 * Weird
 * =====
 *
 * atomicInc: unsigned int
 *  * atomicInc(address,val), computes ((old >= val) ? 0 : (old+1))
 *  * OpenCL equivalent just computes "old + 1", presumably overflowing?
 * atomicDec: unsigned int
 *  * atomicDec(address,val), computes (((old == 0) | (old > val)) ? val : (old-1))
 *  * OpenCL equivalent just computes "old - 1", presumably underflowing?
 * atomicCAS: int, unsigned int, unsigned long long int
 *  * atomicCAS(address,compare,val), computes (old == compare ? val : old)
 * 
 */

}

#endif

#endif
