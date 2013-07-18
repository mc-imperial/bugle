#ifndef OPENCL_ATOMICS_H
#define OPENCL_ATOMICS_H

#ifdef __OPENCL_VERSION__
#define ATOMIC_DECL(OP) \
    _DEVICE_QUALIFIER int __atomic_##OP##_global_int(volatile __global int * x, int y); \
    _DEVICE_QUALIFIER unsigned int __atomic_##OP##_global_unsigned_int(volatile __global unsigned int * x, unsigned int y); \
    _DEVICE_QUALIFIER int __atomic_##OP##_local_int(volatile __local int * x, int y); \
    _DEVICE_QUALIFIER unsigned int __atomic_##OP##_local_unsigned_int(volatile __local unsigned int * x, unsigned int y); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_##OP(volatile __global int * x, int y) { \
      return __atomic_##OP##_global_int(x, y); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned int atomic_##OP(volatile __global unsigned int * x, unsigned int y) { \
      return __atomic_##OP##_global_unsigned_int(x, y); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_##OP(volatile __local int * x, int y) { \
      return __atomic_##OP##_local_int(x, y); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned int atomic_##OP(volatile __local unsigned int * x, unsigned int y) { \
      return __atomic_##OP##_local_unsigned_int(x, y); \
    }

ATOMIC_DECL(add)
ATOMIC_DECL(sub)
ATOMIC_DECL(xchg)
/*ATOMIC_DECL(xchg,float)*/
/*ATOMIC_DECL(cmpxchg)*/
ATOMIC_DECL(min)
ATOMIC_DECL(max)
ATOMIC_DECL(and)
ATOMIC_DECL(or)
ATOMIC_DECL(xor)

#undef ATOMIC_DECL

/* float atomic_xchg(float*,float) */
_DEVICE_QUALIFIER float __atomic_xchg_global_float(volatile __global float * x, float y);
_DEVICE_QUALIFIER float __atomic_xchg_local_float(volatile __local float * x, float y);
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) float atomic_xchg(volatile __global float * x, float y) {
  return __atomic_xchg_global_float(x, y);
}
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) float atomic_xchg(volatile __local float * x, float y) {
  return __atomic_xchg_local_float(x, y);
}

/* TYPE atomic_cmpxchg(TYPE*,TYPE,TYPE) */
/* atomic_cmpxchg(p, cmp, val), computes (old == cmp) ? val : old */

_DEVICE_QUALIFIER int __atomic_cmpxchg_global_int(volatile __global int * x, int y, int z);
_DEVICE_QUALIFIER unsigned int __atomic_cmpxchg_global_unsigned_int(volatile __global unsigned int * x, unsigned int y, unsigned int z);
_DEVICE_QUALIFIER int __atomic_cmpxchg_local_int(volatile __local int * x, int y, int z);
_DEVICE_QUALIFIER unsigned int __atomic_cmpxchg_local_unsigned_int(volatile __local unsigned int * x, unsigned int y, unsigned int z);
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_cmpxchg(volatile __global int * x, int y, int z) {
  return __atomic_cmpxchg_global_int(x, y, z);
}
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned int atomic_cmpxchg(volatile __global unsigned int * x, unsigned int y, unsigned int z) {
  return __atomic_cmpxchg_global_unsigned_int(x, y, z);
}
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_cmpxchg(volatile __local int * x, int y, int z) {
  return __atomic_cmpxchg_local_int(x, y, z);
}
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned int atomic_cmpxchg(volatile __local unsigned int * x, unsigned int y, unsigned int z) {
  return __atomic_cmpxchg_local_unsigned_int(x, y, z);
}

#define ATOMIC_SINGLE_DECL(OP) \
    _DEVICE_QUALIFIER int __atomic_##OP##_global_int(volatile __global int * x); \
    _DEVICE_QUALIFIER unsigned int __atomic_##OP##_global_unsigned_int(volatile __global unsigned int * x); \
    _DEVICE_QUALIFIER int __atomic_##OP##_local_int(volatile __local int * x); \
    _DEVICE_QUALIFIER unsigned int __atomic_##OP##_local_unsigned_int(volatile __local unsigned int * x); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_##OP(volatile __global int * x) { \
      return __atomic_##OP##_global_int(x); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_##OP(volatile __global unsigned int * x) { \
      return __atomic_##OP##_global_unsigned_int(x); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_##OP(volatile __local int * x) { \
      return __atomic_##OP##_local_int(x); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) int atomic_##OP(volatile __local unsigned int * x) { \
      return __atomic_##OP##_local_unsigned_int(x); \
    }

ATOMIC_SINGLE_DECL(inc)
ATOMIC_SINGLE_DECL(dec)

#undef ATOMIC_SINGLE_DECL

#define ATOM_DECL(OP) \
    _DEVICE_QUALIFIER long __atomic_##OP##_global_long(volatile __global long * x, long y); \
    _DEVICE_QUALIFIER unsigned long __atomic_##OP##_global_unsigned_long(volatile __global unsigned long * x, unsigned long y); \
    _DEVICE_QUALIFIER long __atomic_##OP##_local_long(volatile __local long * x, long y); \
    _DEVICE_QUALIFIER unsigned long __atomic_##OP##_local_unsigned_long(volatile __local unsigned long * x, unsigned long y); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atom_##OP(volatile __global long * x, long y) { \
      return __atomic_##OP##_global_long(x, y); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned long atom_##OP(volatile __global unsigned long * x, unsigned long y) { \
      return __atomic_##OP##_global_unsigned_long(x, y); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atom_##OP(volatile __local long * x, long y) { \
      return __atomic_##OP##_local_long(x, y); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned long atom_##OP(volatile __local unsigned long * x, unsigned long y) { \
      return __atomic_##OP##_local_unsigned_long(x, y); \
    }

ATOM_DECL(add)
ATOM_DECL(sub)
ATOM_DECL(xchg)
/*ATOM_DECL(cmpxchg)*/
ATOM_DECL(min)
ATOM_DECL(max)
ATOM_DECL(and)
ATOM_DECL(or)
ATOM_DECL(xor)

#undef ATOM_DECL

/* TYPE atom_cmpxchg(TYPE*,TYPE,TYPE) */
/* atom_cmpxchg(p, cmp, val), computes (old == cmp) ? val : old */

_DEVICE_QUALIFIER long __atom_cmpxchg_global_long(volatile __global long * x, long y, long z);
_DEVICE_QUALIFIER unsigned long __atom_cmpxchg_global_unsigned_long(volatile __global unsigned long * x, unsigned long y, unsigned long z);
_DEVICE_QUALIFIER long __atom_cmpxchg_local_long(volatile __local long * x, long y, long z);
_DEVICE_QUALIFIER unsigned long __atom_cmpxchg_local_unsigned_long(volatile __local unsigned long * x, unsigned long y, unsigned long z);
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atom_cmpxchg(volatile __global long * x, long y, long z) {
  return __atom_cmpxchg_global_long(x, y, z);
}
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned long atom_cmpxchg(volatile __global unsigned long * x, unsigned long y, unsigned long z) {
  return __atom_cmpxchg_global_unsigned_long(x, y, z);
}
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atom_cmpxchg(volatile __local long * x, long y, long z) {
  return __atom_cmpxchg_local_long(x, y, z);
}
_DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) unsigned long atom_cmpxchg(volatile __local unsigned long * x, unsigned long y, unsigned long z) {
  return __atom_cmpxchg_local_unsigned_long(x, y, z);
}

#define ATOM_SINGLE_DECL(OP) \
    _DEVICE_QUALIFIER long __atomic_##OP##_global_long(volatile __global long * x); \
    _DEVICE_QUALIFIER unsigned long __atomic_##OP##_global_unsigned_long(volatile __global unsigned long * x); \
    _DEVICE_QUALIFIER long __atomic_##OP##_local_long(volatile __local long * x); \
    _DEVICE_QUALIFIER unsigned long __atomic_##OP##_local_unsigned_long(volatile __local unsigned long * x); \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atomic_##OP(volatile __global long * x) { \
      return __atomic_##OP##_global_long(x); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atom_##OP(volatile __global unsigned long * x) { \
      return __atomic_##OP##_global_unsigned_long(x); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atom_##OP(volatile __local long * x) { \
      return __atomic_##OP##_local_long(x); \
    } \
    _DEVICE_QUALIFIER static __attribute__((always_inline)) __attribute__((overloadable)) long atom_##OP(volatile __local unsigned long * x) { \
      return __atomic_##OP##_local_unsigned_long(x); \
    }

ATOM_SINGLE_DECL(inc)
ATOM_SINGLE_DECL(dec)

#undef ATOM_SINGLE_DECL
#endif

#endif
