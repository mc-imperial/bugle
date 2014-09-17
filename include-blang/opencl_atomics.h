#ifndef OPENCL_ATOMICS_H
#define OPENCL_ATOMICS_H

typedef __global unsigned int * counter32_t;

#ifdef __OPENCL_VERSION__

#define ATOM_DECL(OP) \
    long __bugle_atomic_##OP##_global_long(volatile __global long * x, long y); \
    unsigned long __bugle_atomic_##OP##_global_unsigned_long(volatile __global unsigned long * x, unsigned long y); \
    long __bugle_atomic_##OP##_local_long(volatile __local long * x, long y); \
    unsigned long __bugle_atomic_##OP##_local_unsigned_long(volatile __local unsigned long * x, unsigned long y); \
    _CLC_INLINE _CLC_OVERLOAD long atom_##OP(volatile __global long * x, long y) { \
      return __bugle_atomic_##OP##_global_long(x, y); \
    } \
    _CLC_INLINE _CLC_OVERLOAD unsigned long atom_##OP(volatile __global unsigned long * x, unsigned long y) { \
      return __bugle_atomic_##OP##_global_unsigned_long(x, y); \
    } \
    _CLC_INLINE _CLC_OVERLOAD long atom_##OP(volatile __local long * x, long y) { \
      return __bugle_atomic_##OP##_local_long(x, y); \
    } \
    _CLC_INLINE _CLC_OVERLOAD unsigned long atom_##OP(volatile __local unsigned long * x, unsigned long y) { \
      return __bugle_atomic_##OP##_local_unsigned_long(x, y); \
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

long __bugle_atomic_cmpxchg_global_long(volatile __global long * x, long y, long z);
unsigned long __bugle_atomic_cmpxchg_global_unsigned_long(volatile __global unsigned long * x, unsigned long y, unsigned long z);
long __bugle_atomic_cmpxchg_local_long(volatile __local long * x, long y, long z);
unsigned long __bugle_atomic_cmpxchg_local_unsigned_long(volatile __local unsigned long * x, unsigned long y, unsigned long z);
_CLC_INLINE _CLC_OVERLOAD long atom_cmpxchg(volatile __global long * x, long y, long z) {
  return __bugle_atomic_cmpxchg_global_long(x, y, z);
}
_CLC_INLINE _CLC_OVERLOAD unsigned long atom_cmpxchg(volatile __global unsigned long * x, unsigned long y, unsigned long z) {
  return __bugle_atomic_cmpxchg_global_unsigned_long(x, y, z);
}
_CLC_INLINE _CLC_OVERLOAD long atom_cmpxchg(volatile __local long * x, long y, long z) {
  return __bugle_atomic_cmpxchg_local_long(x, y, z);
}
_CLC_INLINE _CLC_OVERLOAD unsigned long atom_cmpxchg(volatile __local unsigned long * x, unsigned long y, unsigned long z) {
  return __bugle_atomic_cmpxchg_local_unsigned_long(x, y, z);
}

#define ATOM_SINGLE_DECL(OP) \
    long __bugle_atomic_##OP##_global_long(volatile __global long * x); \
    unsigned long __bugle_atomic_##OP##_global_unsigned_long(volatile __global unsigned long * x); \
    long __bugle_atomic_##OP##_local_long(volatile __local long * x); \
    unsigned long __bugle_atomic_##OP##_local_unsigned_long(volatile __local unsigned long * x); \
    _CLC_INLINE _CLC_OVERLOAD long atom_##OP(volatile __global long * x) { \
      return __bugle_atomic_##OP##_global_long(x); \
    } \
    _CLC_INLINE _CLC_OVERLOAD unsigned long atom_##OP(volatile __global unsigned long * x) { \
      return __bugle_atomic_##OP##_global_unsigned_long(x); \
    } \
    _CLC_INLINE _CLC_OVERLOAD long atom_##OP(volatile __local long * x) { \
      return __bugle_atomic_##OP##_local_long(x); \
    } \
    _CLC_INLINE _CLC_OVERLOAD unsigned long atom_##OP(volatile __local unsigned long * x) { \
      return __bugle_atomic_##OP##_local_unsigned_long(x); \
    }

ATOM_SINGLE_DECL(inc)
ATOM_SINGLE_DECL(dec)

#undef ATOM_SINGLE_DECL
#endif

#endif
