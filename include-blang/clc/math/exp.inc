_CLC_OVERLOAD _CLC_INLINE __CLC_GENTYPE __bugle_exp(__CLC_GENTYPE val) {
  // exp(x) = exp2(x * log2(e))
#if __CLC_FPSIZE == 32
  #define __CLC_M_LOG2E_F 0x1.715476p+0f
  return __bugle_exp_internal_exp2(val * __CLC_M_LOG2E_F);
  #undef __CLC_M_LOG2E_F
#elif __CLC_FPSIZE == 64
  #define __CLC_M_LOG2E 0x1.71547652b82fep+0
  return __bugle_exp_internal_exp2(val * __CLC_M_LOG2E);
  #undef __CLC_M_LOG2E
#else
#error unknown _CLC_FPSIZE
#endif
}
