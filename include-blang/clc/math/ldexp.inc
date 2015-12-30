#ifdef __CLC_SCALAR
#define __CLC_INTN int
#if __CLC_FPSIZE == 32
_CLC_OVERLOAD __CLC_GENTYPE pown(__CLC_GENTYPE x, __CLC_INTN y)
    __asm("llvm.powi.f32");
#else
_CLC_OVERLOAD __CLC_GENTYPE pown(__CLC_GENTYPE x, __CLC_INTN y)
    __asm("llvm.powi.f64");
#endif
#else
#define __bugle_ldexp_pown(x, y) pown(x, y)
#endif

#if __CLC_FPSIZE == 32
#define __CLC_CONST(x) x##f
#else
#define __CLC_CONST(x) x
#endif

_CLC_OVERLOAD __CLC_GENTYPE pown(__CLC_GENTYPE x, __CLC_INTN y);

_CLC_INLINE _CLC_OVERLOAD __CLC_GENTYPE __bugle_ldexp(
    __CLC_GENTYPE x, __CLC_INTN n) {
  return x * pown((__CLC_GENTYPE)__CLC_CONST(2.0), n);
}

#undef __CLC_CONST

#ifdef __CLC_SCALAR
#undef __CLC_INTN
#else

_CLC_INLINE _CLC_OVERLOAD __CLC_GENTYPE __bugle_ldexp(
    __CLC_GENTYPE x, int n) {
  return __bugle_ldexp(x, (__CLC_INTN)n);
}

#endif
